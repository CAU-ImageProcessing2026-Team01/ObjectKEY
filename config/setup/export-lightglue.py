"""Export the pinned official SIFT LightGlue for adaptive native CPU inference.

The C++ loop retains official early stopping and point pruning. Python is needed
only for setup and parity tests, never by the compiled application.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import urllib.request
import torch
import onnx

ROOT=Path(__file__).resolve().parents[2]
COMMIT='5a650c444718caffc2669dc0094cde908f42dd73'
SOURCE_SHA='dcb75b9cad1985c5e7a537c512d4318d47fbcc99df75b39a513de60e027c039f'
WEIGHT_SHA='5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0'
OUT=ROOT/'models/lightglue_sift'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def download(path,url,digest=None):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.download')
        urllib.request.urlretrieve(url,temporary)
        if digest and sha(temporary)!=digest:raise RuntimeError('Downloaded file fingerprint differs: '+str(path))
        temporary.replace(path)
    if digest and sha(path)!=digest:raise RuntimeError('Existing file fingerprint differs: '+str(path))

def official_model():
    path=ROOT/'deps/export/lightglue.py'
    download(path,f'https://raw.githubusercontent.com/cvg/LightGlue/{COMMIT}/lightglue/lightglue.py',SOURCE_SHA)
    download(path.parent/'LICENSE-LightGlue',f'https://raw.githubusercontent.com/cvg/LightGlue/{COMMIT}/LICENSE')
    weight=ROOT/'models/.cache/checkpoints/sift_lightglue_v0-1_arxiv.pth'
    download(weight,'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth',WEIGHT_SHA)
    torch.hub.set_dir(str(ROOT/'models/.cache'))
    spec=importlib.util.spec_from_file_location('official_lightglue',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.LightGlue(features='sift',flash=False,mp=False).eval().cpu()
    state=torch.load(weight,map_location='cpu',weights_only=True)
    for i in range(9):
        state={k.replace(f'self_attn.{i}',f'transformers.{i}.self_attn').replace(f'cross_attn.{i}',f'transformers.{i}.cross_attn'):v for k,v in state.items()}
    state['confidence_thresholds']=model.confidence_thresholds
    model.load_state_dict(state,strict=True)
    return model

class Initialize(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.project=model.input_proj;self.encode=model.posenc
    def forward(self,descriptors,positions):return self.project(descriptors),self.encode(positions)

class Layer(torch.nn.Module):
    def __init__(self,model,index):
        super().__init__();self.layer=model.transformers[index];self.assignment=model.log_assignment[index]
        self.confidence=model.token_confidence[index] if index<8 else None
    def forward(self,desc0,desc1,encoding0,encoding1):
        a,b=self.layer(desc0,desc1,encoding0,encoding1)
        c0,c1=self.confidence(a,b) if self.confidence is not None else (torch.ones_like(a[...,0]),torch.ones_like(b[...,0]))
        z0,z1=self.assignment.matchability(a),self.assignment.matchability(b)
        p0,p1=self.assignment.final_proj(a)/4,self.assignment.final_proj(b)/4
        sim=torch.einsum('bmd,bnd->bmn',p0,p1)
        # ONNX's Log(Sigmoid(z)) export underflows for very negative matchability.
        # -Softplus(-z) is the same log-sigmoid with stable finite CPU evaluation.
        score=torch.log_softmax(sim,2)+torch.log_softmax(sim,1)-torch.nn.functional.softplus(-z0)-torch.nn.functional.softplus(-z1).transpose(1,2)
        return a,b,c0,c1,torch.sigmoid(z0).squeeze(-1),torch.sigmoid(z1).squeeze(-1),score

def main():
    torch.set_num_threads(1);torch.manual_seed(20261122);model=official_model();OUT.mkdir(parents=True,exist_ok=True)
    init=Initialize(model);desc=torch.rand(1,37,128);pos=torch.rand(1,37,4)
    torch.onnx.export(init,(desc,pos),OUT/'initialize.onnx',opset_version=17,input_names=['descriptors','positions'],output_names=['projected','encoding'],
        dynamic_axes={'descriptors':{1:'N'},'positions':{1:'N'},'projected':{1:'N'},'encoding':{3:'N'}},dynamo=False)
    d0,e0=init(desc,pos);d1,e1=init(torch.rand(1,29,128),torch.rand(1,29,4))
    names=['desc0','desc1','encoding0','encoding1'];outputs=['next0','next1','confidence0','confidence1','matchability0','matchability1','scores']
    dynamic={k:{1:'N0' if k.endswith('0') else 'N1'} for k in names[:2]+outputs[:-1]}
    dynamic.update(encoding0={3:'N0'},encoding1={3:'N1'},scores={1:'N0',2:'N1'})
    for i in range(9):
        path=OUT/f'layer{i}.onnx'
        torch.onnx.export(Layer(model,i),(d0,d1,e0,e1),path,opset_version=17,input_names=names,output_names=outputs,dynamic_axes=dynamic,dynamo=False)
        onnx.checker.check_model(str(path));print('Exported',path.name,flush=True)
    files=['initialize.onnx']+[f'layer{i}.onnx' for i in range(9)]
    manifest=['OBJECTKEY_LIGHTGLUE_SIFT_V1',WEIGHT_SHA,'9 0.95 0.99 0.1',*['%s %s'%(f,sha(OUT/f)) for f in files]]
    (OUT/'manifest.txt').write_text('\n'.join(manifest)+'\n',encoding='ascii')
    (OUT/'provenance.json').write_text(json.dumps(dict(commit=COMMIT,source_sha256=SOURCE_SHA,checkpoint_sha256=WEIGHT_SHA,configuration=vars(model.conf),
        export='Layer-wise ONNX17; native adaptive depth/width, official RootSIFT, SIFT scales/orientations and CPU settings.',files={f:sha(OUT/f) for f in files}),indent=2))
    download(OUT/'LICENSE-LightGlue',f'https://raw.githubusercontent.com/cvg/LightGlue/{COMMIT}/LICENSE')
    print('LightGlue export complete.',flush=True)

if __name__=='__main__':main()
