package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Writer turns per-cgroup feature maps into output records: CSV/JSONL files plus
// an optional POST to the detector's /detect/batch.
type Writer struct {
	dir     string
	format  string
	node    string
	pushURL string
	mapper  *Mapper
	client  *http.Client
}

func NewWriter(dir, format, node, pushURL string, mapper *Mapper) *Writer {
	if pushURL != "" {
		pushURL = strings.TrimRight(pushURL, "/")
	}
	return &Writer{dir: dir, format: format, node: node, pushURL: pushURL,
		mapper: mapper, client: &http.Client{Timeout: 5 * time.Second}}
}

type record struct {
	Timestamp string
	Namespace string
	Pod       string
	Container string
	Features  map[string]float64
}

func (w *Writer) build(perCg map[uint64]map[string]float64) []record {
	ts := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	recs := make([]record, 0, len(perCg))
	for cg, feats := range perCg {
		ns, pod, cname := "unknown", fmt.Sprintf("cg-%d", cg), "unknown"
		if w.mapper != nil {
			id := w.mapper.Resolve(cg)
			ns, pod, cname = id.Namespace, id.Pod, id.Container
		}
		full := make(map[string]float64, len(featureOrder))
		for _, f := range featureOrder {
			full[f] = feats[f] // zero value if absent
		}
		recs = append(recs, record{ts, ns, pod, cname, full})
	}
	return recs
}

// Emit writes this window's records and pushes them to the detector.
func (w *Writer) Emit(perCg map[uint64]map[string]float64) {
	recs := w.build(perCg)
	if len(recs) == 0 {
		return
	}
	if w.format == "csv" || w.format == "both" {
		if err := w.writeCSV(recs); err != nil {
			log.Printf("[warn] csv write: %v", err)
		}
	}
	if w.format == "jsonl" || w.format == "both" {
		if err := w.writeJSONL(recs); err != nil {
			log.Printf("[warn] jsonl write: %v", err)
		}
	}
	w.push(recs)
	log.Printf("[collector-go] window emitted %d feature rows -> %s", len(recs), w.dir)
}

func (w *Writer) header() []string {
	return append([]string{"timestamp", "node", "namespace", "pod", "container"}, featureOrder...)
}

func (w *Writer) row(r record) []string {
	row := []string{r.Timestamp, w.node, r.Namespace, r.Pod, r.Container}
	for _, f := range featureOrder {
		row = append(row, strconv.FormatFloat(r.Features[f], 'g', -1, 64))
	}
	return row
}

func (w *Writer) writeCSV(recs []record) error {
	path := filepath.Join(w.dir, "features.csv")
	_, statErr := os.Stat(path)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	cw := csv.NewWriter(f)
	if os.IsNotExist(statErr) {
		cw.Write(w.header())
	}
	for _, r := range recs {
		cw.Write(w.row(r))
	}
	cw.Flush()
	return cw.Error()
}

func (w *Writer) writeJSONL(recs []record) error {
	path := filepath.Join(w.dir, "features.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, r := range recs {
		obj := w.item(r)
		b, _ := json.Marshal(obj)
		f.Write(append(b, '\n'))
	}
	return nil
}

// item is the per-row JSON object: flat feature keys + identity, matching the
// detector's FeatureVector (extra=allow, flat feature keys accepted).
func (w *Writer) item(r record) map[string]any {
	o := map[string]any{
		"timestamp": r.Timestamp, "node": w.node,
		"namespace": r.Namespace, "pod": r.Pod, "container": r.Container,
	}
	for _, f := range featureOrder {
		o[f] = r.Features[f]
	}
	return o
}

func (w *Writer) push(recs []record) {
	if w.pushURL == "" {
		return
	}
	items := make([]map[string]any, 0, len(recs))
	for _, r := range recs {
		items = append(items, w.item(r))
	}
	body, _ := json.Marshal(map[string]any{"items": items})
	url := w.pushURL + "/detect/batch"
	resp, err := w.client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("[collector-go] push to %s failed: %v", url, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		log.Printf("[collector-go] push to %s -> HTTP %d", url, resp.StatusCode)
	}
}
