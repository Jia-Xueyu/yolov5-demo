from ruamel import yaml

#[from, number, module, arg]
# nc=80
# backbone_list=[]
# backbone_module_list=['Focus','Conv','BottleneckCSP','Conv','BottleneckCSP','Conv','BottleneckCSP','Conv','SPP','BottleneckCSP']
# save_backbone_level=[4,6,8]
# backbone_list.append([-1,1,backbone_module_list[0],[64,3]])
# backbone_list.append([-1,1,backbone_module_list[1],[128,3,2]])
# backbone_list.append([-1,3,backbone_module_list[2],[128]])
# backbone_list.append([-1,1,backbone_module_list[3],[256,3,2]])
# backbone_list.append([-1,9,backbone_module_list[4],[256]])
# backbone_list.append([-1,1,backbone_module_list[5],[512,3,2]])
# backbone_list.append([-1,9,backbone_module_list[6],[512]])
# backbone_list.append([-1,1,backbone_module_list[7],[1024,3,2]])
# backbone_list.append([-1,1,backbone_module_list[8],[1024,[5,9,13]]])
# backbone_list.append([-1,3,backbone_module_list[9],[1024]])
nc=80
anchors=[[10,13, 16,20, 33,23],
     [30,61, 62,45, 59,119],
     [116,90, 156,198, 373,326]]
cfg={
    'nc':nc,
    'anchors':anchors,
    'backbone':
    [[-1,1,'Focus',[64,3]],#0
    [-1,1,'Conv',[128,3,2]],
    [-1,3,'BottleneckCSP',[128]],
    [-1,1,'Conv',[256,3,2]],
    [-1,9,'BottleneckCSP',[256]],
    [-1,1,'Conv',[512,3,2]],
    [-1,9,'BottleneckCSP',[512]],
    [-1,1,'Conv',[1024,3,2]],
    [-1,1,'SPP',[1024,[5,9,13]]],
    [-1,3,'BottleneckCSP',[1024]]],#9
    'head':
    [[-1,1,'Conv',[512,1,1]],#10
     [-1,1,'nn.Upsample',[None,2,'nearest']],
     [[-1,6],1,'Concat',[1]],
     [-1,3,'BottleneckCSP',[512]],
     [-1,1,'Conv',[256,1,1]],
     [-1,1,'nn.Upsample',[None,2,'nearest']],
     [[-1,4],1,'Concat',[1]],
     [-1,3,'BottleneckCSP',[256]],#17

     [-1,1,'Conv',[256,3,2]],
     [[-1,14],1,'Concat',[1]],
     [-1,3,'BottleneckCSP',[512]],#20

     [-1,1,'Conv',[512,3,2]],
     [[-1,10],1,'Concat',[1]],
     [-1,3,'BottleneckCSP',[1024]],#23

     [[17,20,23],1,'Detect',[nc,anchors]]],

}

# head_list=[]
# head_module_list=['Conv','nn.Upsample','Concat','BottleneckCSP','Conv','nn.Upsample','Concat','BottleneckCSP','Conv','Concat','BottleneckCSP','Conv','Concat','BottleneckCSP']
# save_head_level=[17,20,23]
# cat_level=[10,14]
#
# head_list.append([-1,1,head_module_list[0],[512,1,1]])
# head_list.append([-1,1,head_module_list[1],[None,2,'nearest']])
# head_list.append([[-1,6],1,head_module_list[2],[1]])
# head_list.append([-1,3,head_module_list[3],[512]])
# head_list.append([-1,1,head_module_list[4],[256,1,1]])
# head_list.append([-1,1,head_module_list[5],[None,2,'nearest']])
# head_list.append([[-1,4],1,head_module_list[6],[1]])
# head_list.append([-1,3,head_module_list[7],[256]])
#
# head_list.append([-1,1,head_module_list[8],[256,3,2]])
# head_list.append([[-1,14],1,head_module_list[9],[1]])
# head_list.append([-1,3,head_module_list[10],[512]])
#
# head_list.append([-1,1,head_module_list[11],[512,3,2]])
# head_list.append([[-1,10],1,head_module_list[12],[1]])
# head_list.append([-1,3,head_module_list[13],[1024]])
#
# model={'nc':nc,'backbone':backbone_list,'head':head_list}

with open('./model.yaml','w') as f:
    yaml.dump(cfg,f,Dumper=yaml.RoundTripDumper)