"""Experimental D3 fixed-shape captured executor; no token-time shape switch."""
from __future__ import annotations
from typing import Any
import torch
from freetoken.kernel import moe_sum_reduce_triton
D3_EXECUTOR_SCHEMA="freetoken.inferswarm-d3-graph-multiworker/1"
D3_TOPOLOGY="unified_three_device_whole_model_graph_independent_ab_fanout"
D3_DEPENDENCY="cuda_capture_internal_gpu0_ready_ab_independent_done_fanin"
D3_FANOUT_SHAPE="CONCURRENT_BOUNDED_TWO_WORKER"
def _active(value):
    value=tuple(value)
    if value not in (("a",),("b",),("a","b")): raise ValueError("D3 active workers must be a, b, or ab")
    return value
def build_d3_route_lookups(p, device, active_workers=("a","b")):
    active=_active(active_workers); n,e=p.worker_a.num_layers,p.worker_a.num_experts; aset,bset,local=set(p.worker_a.flat_ids_in_rank_order),set(p.worker_b.flat_ids_in_rank_order),set(p.local_remainder)
    if aset&bset or len(aset|bset)!=6000 or len(local)!=4240 or (aset|bset)&local or aset|bset|local!=set(range(n*e)): raise ValueError("D3 frozen placement does not resolve every identity exactly once")
    out=[]
    for label,w in (("a",p.worker_a),("b",p.worker_b)):
        table=None
        if label in active:
            table=torch.full((n,e),-1,dtype=torch.int32,device=device)
            for layer in w.per_layer:
                if layer.expert_ids: table[layer.layer_id,torch.tensor(layer.expert_ids,device=device)]=torch.tensor(layer.remote_slots,dtype=torch.int32,device=device)
        out.append(table)
    return tuple(out)
def build_d3_local_fallback_ids(p,device,active_workers=("a","b")):
    active=_active(active_workers); n,e=p.worker_a.num_layers,p.worker_a.num_experts; local=set(p.local_remainder)
    if "a" not in active: local.update(p.worker_a.flat_ids_in_rank_order)
    if "b" not in active: local.update(p.worker_b.flat_ids_in_rank_order)
    values=[]
    for layer in range(n):
        x=next((expert for expert in range(e) if layer*e+expert in local),None)
        if x is None: raise ValueError(f"D3 layer {layer} has no runtime-GPU0 fallback identity")
        values.append(x)
    return torch.tensor(values,dtype=torch.int32,device=device)
def validate_d3_runtime(config,cache,banks,workers,active_workers=("a","b")):
    active=_active(active_workers)
    if bool(getattr(config,"inferswarm_remote_decode",False)) or bool(getattr(config,"inferswarm_experimental_d2_graph_remote",False)): raise ValueError("D3 is mutually exclusive with canonical remote decode and D2")
    if int(getattr(getattr(config,"tp_info",None),"size",-1))!=1 or int(getattr(config,"cuda_graph_max_bs",-1) or -1)!=1 or int(getattr(config,"max_running_req",-1))!=1: raise ValueError("D3 requires TP1, CUDA graph BS1, and max running requests 1")
    if getattr(config,"moe_backend",None)!="offload" or getattr(cache,"decode_target",None)!="gpu" or getattr(cache,"cpu_layer_ids",frozenset()) or getattr(cache,"quant_format",None)!="nvfp4": raise ValueError("D3 requires GPU decode/offload/native NVFP4")
    for label,bank,worker in (("a",banks.worker_a,workers[0]),("b",banks.worker_b,workers[1])):
        if (label in active)!=(bank is not None and worker is not None): raise ValueError("D3 active resident worker binding disagreement")
        if bank and bank.report.secondary_visible_ordinal!=worker.secondary.visible_ordinal: raise ValueError("D3 resident bank device binding disagreement")
    if len(active)==2 and workers[0].secondary.visible_ordinal==workers[1].secondary.visible_ordinal: raise ValueError("D3 worker devices must be distinct")
class InferSwarmD3GraphMultiworkerExecutor:
 def __init__(self,*,resident_banks,worker_a_device,worker_b_device,primary_device,worker_a_slot_lookup,worker_b_slot_lookup,local_fallback_ids,active_workers=("a","b"),hidden_size,top_k,hidden_dtype,num_layers,intermediate_size,torch_module=torch):
  self.active_workers=_active(active_workers);self.resident_banks=resident_banks;self.primary_device=primary_device;self.worker_a_device=worker_a_device;self.worker_b_device=worker_b_device;self.worker_a_slot_lookup=worker_a_slot_lookup;self.worker_b_slot_lookup=worker_b_slot_lookup;self.local_fallback_ids=local_fallback_ids;self.hidden_size,self.top_k,self.hidden_dtype,self.num_layers,self.intermediate_size=int(hidden_size),int(top_k),hidden_dtype,int(num_layers),int(intermediate_size);self._torch,self._captured_bs,self._capture_complete,self._failure_count,self._steady_state_host_sync_count,self._graph_recapture_count=torch_module,(),False,0,0,0
  cuda,primary=torch_module.cuda,int(primary_device.index)
  try:
   cuda.set_device(primary);self.host_activation=torch_module.empty((1,self.hidden_size),dtype=hidden_dtype,pin_memory=True);self.gpu0_local_ids=torch_module.empty((1,self.top_k),dtype=torch_module.int32,device=primary_device);self.gpu0_local_weights=torch_module.empty((1,self.top_k),dtype=torch_module.float32,device=primary_device);self.gpu0_local_routes=torch_module.empty((1,self.top_k,self.hidden_size),dtype=hidden_dtype,device=primary_device);self.gpu0_reconstruction=torch_module.empty_like(self.gpu0_local_routes);self.gpu0_output=torch_module.empty((1,self.hidden_size),dtype=hidden_dtype,device=primary_device);self.gpu0_gate_up=torch_module.empty((1,self.top_k,2*self.intermediate_size),dtype=hidden_dtype,device=primary_device);self.gpu0_activation_out=torch_module.empty((self.top_k,self.intermediate_size),dtype=hidden_dtype,device=primary_device);self.zero_weight=torch_module.zeros((),dtype=torch_module.float32,device=primary_device);self.device_counts=torch_module.zeros((self.num_layers,5),dtype=torch_module.int64,device=primary_device);self.ready_events=[cuda.Event() for _ in range(self.num_layers)]
   for x in self.active_workers:
    d=torch.device("cuda",int(getattr(self,f"worker_{x}_device").secondary.visible_ordinal));setattr(self,f"worker_{x}_torch_device",d);setattr(self,f"worker_{x}_stream",cuda.Stream(device=d));setattr(self,f"done_{x}_events",[cuda.Event() for _ in range(self.num_layers)])
    for name,shape,dtype,dev in (("slots",(1,self.top_k),torch_module.int32,primary_device),("weights",(1,self.top_k),torch_module.float32,primary_device),("return_routes",(1,self.top_k,self.hidden_size),hidden_dtype,primary_device)): setattr(self,f"gpu0_{x}_{name}",torch_module.empty(shape,dtype=dtype,device=dev))
    for name,shape,dtype in (("slots",(1,self.top_k),torch_module.int32),("weights",(1,self.top_k),torch_module.float32),("return",(1,self.top_k,self.hidden_size),hidden_dtype)): setattr(self,f"host_{x}_{name}",torch_module.empty(shape,dtype=dtype,pin_memory=True))
    for name,shape,dtype in (("activation",(1,self.hidden_size),hidden_dtype),("slots",(1,self.top_k),torch_module.int32),("weights",(1,self.top_k),torch_module.float32),("routes",(1,self.top_k,self.hidden_size),hidden_dtype),("gate_up",(1,self.top_k,2*self.intermediate_size),hidden_dtype),("activation_out",(self.top_k,self.intermediate_size),hidden_dtype)): setattr(self,f"worker_{x}_{name}",torch_module.empty(shape,dtype=dtype,device=d))
  finally: cuda.set_device(primary)
  self.timing_hooks={f"worker_{x}_branch":(f"worker_{x}_local_start",f"worker_{x}_local_end") for x in self.active_workers}
 def _validate(self,l,h,w,i):
  n=int(l.layer_id)
  if not 0<=n<self.num_layers or tuple(h.shape)!=(1,self.hidden_size) or tuple(w.shape)!=(1,self.top_k) or tuple(i.shape)!=(1,self.top_k):raise RuntimeError("D3 accepts only captured batch-one routing geometry")
  if h.device!=self.primary_device or w.device!=self.primary_device or i.device!=self.primary_device or h.dtype!=self.hidden_dtype or w.dtype!=torch.float32 or i.dtype!=torch.int32:raise RuntimeError("D3 inputs disagree with the captured primary-device contract")
  return n
 def _worker_branch(self,x,l,c,n):
  cuda=self._torch.cuda;d=getattr(self,f"worker_{x}_torch_device");s=getattr(self,f"worker_{x}_stream");cuda.set_device(d)
  with cuda.stream(s):
   self.ready_events[n].wait(s);getattr(self,f"worker_{x}_activation").copy_(self.host_activation,non_blocking=True);getattr(self,f"worker_{x}_slots").copy_(getattr(self,f"host_{x}_slots"),non_blocking=True);getattr(self,f"worker_{x}_weights").copy_(getattr(self,f"host_{x}_weights"),non_blocking=True);bank=getattr(self.resident_banks,f"worker_{x}");l._expert_route_contributions(c,getattr(self,f"worker_{x}_activation"),getattr(self,f"worker_{x}_weights"),getattr(self,f"worker_{x}_slots"),views=bank.bank_views(),alphas=bank.alpha_views(),out=getattr(self,f"worker_{x}_routes"),gate_up_out=getattr(self,f"worker_{x}_gate_up"),activation_out=getattr(self,f"worker_{x}_activation_out"));getattr(self,f"host_{x}_return").copy_(getattr(self,f"worker_{x}_routes"),non_blocking=True);getattr(self,f"done_{x}_events")[n].record(s)
 def decode(self,l,c,h,w,ids):
  n=self._validate(l,h,w,ids);cuda,primary=self._torch.cuda,int(self.primary_device.index)
  try:
   cuda.set_device(primary);s=cuda.current_stream(self.primary_device);masks={}
   for x in self.active_workers:
    slots=getattr(self,f"worker_{x}_slot_lookup")[n][ids.long()];getattr(self,f"gpu0_{x}_slots").copy_(slots);masks[x]=getattr(self,f"gpu0_{x}_slots")>=0;torch.where(masks[x],w,self.zero_weight,out=getattr(self,f"gpu0_{x}_weights"));getattr(self,f"gpu0_{x}_slots").clamp_min_(0);getattr(self,f"host_{x}_slots").copy_(getattr(self,f"gpu0_{x}_slots"),non_blocking=True);getattr(self,f"host_{x}_weights").copy_(getattr(self,f"gpu0_{x}_weights"),non_blocking=True)
   local=~masks[self.active_workers[0]] if len(self.active_workers)==1 else ~(masks["a"]|masks["b"]);torch.where(local,ids,self.local_fallback_ids[n],out=self.gpu0_local_ids);torch.where(local,w,self.zero_weight,out=self.gpu0_local_weights);self.device_counts[n,0].add_(self.top_k);self.device_counts[n,1].add_(masks["a"].sum() if "a" in masks else 0);self.device_counts[n,2].add_(masks["b"].sum() if "b" in masks else 0);self.device_counts[n,3].add_(local.sum());self.device_counts[n,4].add_(1);self.host_activation.copy_(h,non_blocking=True);self.ready_events[n].record(s)
   for x in self.active_workers:self._worker_branch(x,l,c,n)
   cuda.set_device(primary);c.ensure_experts(n,self.gpu0_local_ids);c.copy_missing();l._expert_route_contributions(c,h,self.gpu0_local_weights,self.gpu0_local_ids,views=c.bank_views(),alphas=c.alphas_for_slots(n),out=self.gpu0_local_routes,gate_up_out=self.gpu0_gate_up,activation_out=self.gpu0_activation_out);routes=self.gpu0_local_routes
   for x in self.active_workers:getattr(self,f"done_{x}_events")[n].wait(s);getattr(self,f"gpu0_{x}_return_routes").copy_(getattr(self,f"host_{x}_return"),non_blocking=True);torch.add(routes,getattr(self,f"gpu0_{x}_return_routes"),out=self.gpu0_reconstruction);routes=self.gpu0_reconstruction
   moe_sum_reduce_triton(routes,self.gpu0_output);return self.gpu0_output
  except Exception:
   self._failure_count+=1
   try:
    for x in self.active_workers:cuda.set_device(getattr(self,f"worker_{x}_torch_device"));getattr(self,f"worker_{x}_stream").synchronize()
   finally:cuda.set_device(primary)
   raise
  finally:cuda.set_device(primary)
 def set_graph_state(self,captured_bs):
  v=tuple(sorted(int(x) for x in captured_bs))
  if v!=(1,):raise RuntimeError(f"D3 refused silent eager fallback: expected exactly CUDA graph BS1, captured {list(v)}")
  if self._capture_complete:self._graph_recapture_count+=1
  self._captured_bs,self._capture_complete=v,True;self.reset_counters()
 def reset_counters(self):self.device_counts.zero_()
 def configuration_report(self)->dict[str,Any]:
  a,b=self.resident_banks.worker_a,self.resident_banks.worker_b;primary=(self.worker_a_device or self.worker_b_device).primary;active=self._capture_complete and self._captured_bs==(1,)
  return {"schema":D3_EXECUTOR_SCHEMA,"experimental":True,"enabled":True,"active_workers":list(self.active_workers),"active_worker_count":len(self.active_workers),"worker_a_active":"a" in self.active_workers,"worker_b_active":"b" in self.active_workers,"no_inactive_worker_allocation_or_branch":True,"graph_active":active,"graph_topology":"gpu0_local_plus_"+"_and_".join(self.active_workers)+"_fanin","captured_batch_sizes":list(self._captured_bs),"graph_replays_per_token":1 if active else 0,"graph_recapture_count":self._graph_recapture_count,"eager_fallback":not active,"cross_device_dependency":D3_DEPENDENCY,"fanout_shape":D3_FANOUT_SHAPE if len(self.active_workers)==2 else "CAPTURED_SINGLE_WORKER","primary_uuid":primary.uuid,"worker_a_uuid":self.worker_a_device.secondary.uuid if self.worker_a_device else None,"worker_b_uuid":self.worker_b_device.secondary.uuid if self.worker_b_device else None,"corrected_placement_sha256":(a or b).placement.artifact_sha256,"worker_a_resident_slots":a.placement.remote_slots if a else 0,"worker_b_resident_slots":b.placement.remote_slots if b else 0,"total_active_resident_slots":sum(x.placement.remote_slots for x in (a,b) if x),"total_active_resident_expert_bytes":self.resident_banks.total_native_expert_bytes,"steady_state_expert_weight_bytes_host_to_worker_a":0,"steady_state_expert_weight_bytes_host_to_worker_b":0,"steady_state_host_sync_count":self._steady_state_host_sync_count,"fixed_allocations":True,"stable_tensor_addresses":True,"reconstruction_method":"elementwise_local_plus_active_worker_routes_then_one_canonical_route_sum","timing_instrumentation":{"bounded":True,"hooks":self.timing_hooks}}
 def snapshot(self):
  z=self.device_counts.detach().cpu();total,a,b,local,calls=(int(z[:,i].sum().item()) for i in range(5));exact=total==a+b+local
  return {**self.configuration_report(),"ownership":{"total_router_selections":total,"executed_on_worker_a":a,"executed_on_worker_b":b,"executed_on_gpu0_local":local,"layer_calls":calls,"selection_arithmetic_exact":exact,"no_route_dropped":exact,"no_route_duplicated":exact,"worker_ab_disjoint":True}}
def absent_d3_graph_multiworker_report():return {"schema":D3_EXECUTOR_SCHEMA,"experimental":True,"enabled":False,"active_workers":[],"active_worker_count":0,"graph_active":False,"graph_topology":None,"captured_batch_sizes":[],"graph_replays_per_token":0,"eager_fallback":False,"steady_state_host_sync_count":0}
