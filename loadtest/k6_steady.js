import http from "k6/http";
import { check } from "k6";
const BATCH=30; const URL=__ENV.TARGET||"http://ebpf-detector.ebpf-final:8080/detect/batch";
const F=["syscall_read_rate","syscall_write_rate","syscall_open_rate","tcp_connect_rate","tcp_retransmit_rate","network_rx_bytes","network_tx_bytes","disk_read_bytes","disk_write_bytes","disk_io_latency_ms","process_exec_count","process_fork_count","context_switch_count","cpu_utilization"];
export const options={scenarios:{steady:{executor:"constant-arrival-rate",rate:Number(__ENV.RATE||15),timeUnit:"1s",duration:"60s",preAllocatedVUs:50,maxVUs:200}},thresholds:{http_req_duration:["p(99)<1000","p(95)<500"],http_req_failed:["rate<0.01"]}};
function vec(i){const v={timestamp:new Date().toISOString(),node:"lt",namespace:"bench",pod:"b"+i,container:"c"};for(const f of F)v[f]=Number((Math.random()*50).toFixed(3));return v;}
export default function(){const items=[];for(let i=0;i<BATCH;i++)items.push(vec(i));const r=http.post(URL,JSON.stringify({items}),{headers:{"Content-Type":"application/json"},timeout:"10s"});check(r,{"200":x=>x.status===200});}
