import torch
import torch.nn as nn
from InstanceDiffusion.ldm.modules.diffusionmodules.util import FourierEmbedder


class UniFusion(nn.Module):
    def __init__(self,  in_dim, out_dim, mid_dim=3072, fourier_freqs=8, 
    train_add_boxes=True, train_add_points=False, train_add_scribbles=False, train_add_masks=False, 
    test_drop_boxes=False, test_drop_points=False, test_drop_scribbles=False, test_drop_masks=False,
    use_seperate_tokenizer=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.mid_dim = mid_dim

        # InstanceDiffusion hyper-parameters
        # BBOX-ONLY: other modality constants disabled
        # self.n_scribble_points = 20
        # self.n_polygon_points = 256
        fourier_freqs = 16
        # fourier_freqs_polygons = 16
        # Always enable boxes in bbox-only mode
        self.add_boxes = True
        # Force bbox-only: ignore other modality flags even if passed
        self.add_points = False
        self.add_scribbles = False
        self.add_masks = False
        self.use_seperate_tokenizer = use_seperate_tokenizer

        # Use instance masks as additional model inputs for mask conditioned image generation
        # With bbox-only, never use segmentation tokens/backbone
        self.use_segs = False

        # if self.use_segs:
        #     in_dim = 30
        #     self.resize_input = 512
        #     self.down_factor = 64 # determined by the convnext backbone
        #     self.in_conv = nn.Conv2d(in_dim,3,3,1,1) # from num_sem to 3 channels
        #     self.convnext_tiny_backbone = convnext_tiny(pretrained=True)
        #     self.num_tokens = (self.resize_input // self.down_factor) ** 2
        #     self.convnext_feature_dim = 3072
        #     self.pos_embedding = nn.Parameter(torch.empty(1, self.num_tokens, self.convnext_feature_dim).normal_(std=0.02))  # from BERT

        self.test_drop_boxes = test_drop_boxes
        # BBOX-ONLY: set other modality drop flags to False
        self.test_drop_points = False
        self.test_drop_scribbles = False
        self.test_drop_masks = False
        self.test_drop_segs = False

        # Training-time random box-drop ratio (disabled by default for our use).
        # Upstream sometimes uses 0.1 to mimic CFG; we keep 0.0 so boxes are
        # always available during training unless the dataset truly lacks them.
        self.drop_box_ratio = 0.0


        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        # self.fourier_embedder_polygons = FourierEmbedder(num_freqs=fourier_freqs_polygons)
        input_dim = self.in_dim
        input_dim_list = []
        if self.add_boxes:
            self.position_dim = fourier_freqs*2*4 # 2: sin and cos; 4: (x1,y1) and (x2,y2)
            input_dim += self.position_dim
            input_dim_list.append(self.in_dim+self.position_dim)
        # BBOX-ONLY: remove additional modality branches from tokenizer input
        # if self.add_points:
        #     self.point_dim = fourier_freqs*2*2
        #     input_dim += self.point_dim
        #     input_dim_list.append(self.in_dim+self.point_dim)
        # if self.add_scribbles:
        #     self.scribble_dim = fourier_freqs_polygons*2*self.n_scribble_points*2
        #     input_dim += self.scribble_dim
        #     input_dim_list.append(self.in_dim+self.scribble_dim)
        # if self.add_masks:
        #     self.polygon_dim = fourier_freqs_polygons*2*self.n_polygon_points*2
        #     input_dim += self.polygon_dim
        #     input_dim_list.append(self.in_dim+self.polygon_dim)
        #     if self.use_segs:
        #         input_dim += self.convnext_feature_dim
        #         input_dim_list.append(self.convnext_feature_dim)

        if self.use_seperate_tokenizer:
            self.linears_list = nn.ModuleList([])
            for idx, input_dim_ in enumerate(input_dim_list):
                mid_dim = self.mid_dim
                self.linears_list.append(nn.Sequential(
                    nn.Linear( input_dim_, mid_dim),
                    nn.SiLU(),
                    nn.Linear( mid_dim, mid_dim),
                    nn.SiLU(),
                    nn.Linear(mid_dim, out_dim),
                ))
        else:
            self.linears = nn.Sequential(
                nn.Linear( input_dim, self.mid_dim),
                nn.SiLU(),
                nn.Linear( self.mid_dim, self.mid_dim),
                nn.SiLU(),
                nn.Linear(self.mid_dim, out_dim),
            )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim])) # text
        if self.add_boxes:
            self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
        # BBOX-ONLY: no learnable nulls for other modalities
        # if self.add_points:
        #     self.null_point_feature = torch.nn.Parameter(torch.zeros([self.point_dim]))
        # if self.add_scribbles:
        #     self.null_scribble_feature = torch.nn.Parameter(torch.zeros([self.scribble_dim]))
        # if self.add_masks:
        #     self.null_polygon_feature = torch.nn.Parameter(torch.zeros([self.polygon_dim]))
        #     if self.use_segs:
        #         self.null_seg_feature = torch.nn.Parameter(torch.zeros([self.convnext_feature_dim]))

    def reset_dropout_test(self):
        # BBOX-ONLY: simplified test dropout (only drop_box is used)
        drop_box = self.test_drop_boxes
        # drop_point = self.test_drop_points
        # drop_scribble = self.test_drop_scribbles
        # drop_polygons = self.test_drop_masks
        # drop_segs = self.test_drop_masks
        # return drop_point, drop_box, drop_scribble, drop_polygons, drop_segs
        # BBOX-ONLY: non-box modalities are always considered dropped (True)
        return True, drop_box, True, True, True

    def reset_dropout(self):
        drop_box = False
        drop_point = False
        drop_scribble = False
        drop_polygons = False
        drop_segs = False
        return drop_point, drop_box, drop_scribble, drop_polygons, drop_segs

    def reset_dropout_train(self, drop_point, drop_box, drop_scribble, drop_polygons, drop_segs):
        if not drop_polygons:
            drop_box = False
            drop_point = False
        if not drop_box or not drop_polygons:
            drop_point = False

        # keep point only for 10% of the time
        keep_point_only_ratio = 0.1
        keep_point_only = torch.rand(1).item() < keep_point_only_ratio
        if keep_point_only:
            drop_point = False
            drop_box = True
            drop_scribble = True
            drop_polygons = True
            drop_segs = True

        # keep scribble only for 0% of the time
        keep_scribble_only_ratio = 0.0
        keep_scribble_only = torch.rand(1).item() < keep_scribble_only_ratio and not drop_scribble
        if keep_scribble_only:
            drop_point = True
            drop_box = True
            drop_scribble = False
            drop_polygons = True
            drop_segs = True

        # keep mask only for 0% of the time
        keep_mask_only_ratio = 0.0
        keep_mask_only = torch.rand(1).item() < keep_mask_only_ratio and not drop_polygons
        if keep_mask_only:
            drop_point = True
            drop_box = True
            drop_scribble = True
            drop_polygons = False
            drop_segs = False

        # keep seg only for 10% of the time
        keep_seg_only_ratio = 0.1 # default 0.1
        keep_seg_only = torch.rand(1).item() < keep_seg_only_ratio and not drop_segs
        if keep_seg_only:
            drop_point = False
            drop_box = False
            drop_scribble = True
            drop_polygons = False
            drop_segs = False

        # keep box only for 0% of the time
        keep_box_only_ratio = 0.0 # default 0.0
        keep_box_only = torch.rand(1).item() < keep_box_only_ratio and not drop_box
        if keep_box_only:
            drop_point = True
            drop_box = False
            drop_scribble = True
            drop_polygons = True
            drop_segs = True

        return drop_point, drop_box, drop_scribble, drop_polygons, drop_segs

    def forward(self, boxes, masks, positive_embeddings, scribbles=None, polygons=None, segs=None, points=None):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        drop_point, drop_box, drop_scribble, drop_polygons, drop_segs = self.reset_dropout()
        # randomly drop boxes or points embeddings.
        if self.add_boxes:
            # Original upstream behavior (kept for reference):
            # drop_box_ratio = 0.1
            # drop_box = torch.rand(1).item() < drop_box_ratio
            drop_box_ratio = float(getattr(self, 'drop_box_ratio', 0.0))
            drop_box = drop_box_ratio > 0.0 and torch.rand(1).item() < drop_box_ratio
        # if self.add_points:
        #     drop_point_ratio = 0.1
        #     drop_point = torch.rand(1).item() < drop_point_ratio
        # if self.add_scribbles:
        #     drop_scribble_ratio = 0.1
        #     drop_scribble = torch.rand(1).item() < drop_scribble_ratio
        # if self.add_masks:
        #     drop_polygon_ratio = 0.1
        #     drop_polygons = torch.rand(1).item() < drop_polygon_ratio
        #     drop_segs = drop_polygons

        # Not training, always keep both boxes and points
        if not self.training:
            drop_point, drop_box, drop_scribble, drop_polygons, drop_segs = self.reset_dropout_test()
        else:
            # BBOX-ONLY: simplify train-time dropout
            # Mark non-box modalities dropped (True) so the guard below can ensure boxes stay enabled.
            drop_point, drop_box, drop_scribble, drop_polygons, drop_segs = True, drop_box, True, True, True

        # set drop_box to False if all other inputs are dropped
        if drop_point and drop_box and drop_scribble and drop_polygons and drop_segs:
            drop_box = False

        # embedding position (it may includes padding as placeholder)
        if self.add_boxes:
            xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C (C=8*2*4)
        # if self.add_points:
        #     if points is None: # we can always get a point using a box
        #         points = (boxes[:, :, :2] + boxes[:, :, 2:]) / 2.0
        #     point_embedding = self.fourier_embedder(points) # B*N*2 --> B*N*(8*2*2)
        # if self.add_scribbles:
        #     scribble_embedding = self.fourier_embedder_polygons(scribbles) # B*N*20 --> B*N*(8*20*2)
        # if self.add_masks:
        #     polygon_embedding = self.fourier_embedder_polygons(polygons) # B*N*128 --> B*N*(16*128*2)
        # if self.use_segs:
        #     segs = torch.nn.functional.interpolate(segs, self.resize_input, mode="nearest")
        #     segs_feature = self.in_conv(segs)
        #     segs_feature = self.convnext_tiny_backbone(segs_feature)
        #     segs_feature = segs_feature.reshape(B, -1, self.num_tokens)
        #     segs_feature = segs_feature.permute(0, 2, 1)
            
        # learnable null embedding
        positive_null = self.null_positive_feature.view(1,1,-1).to(positive_embeddings.device)
        if self.add_boxes:
            xyxy_null = self.null_position_feature.view(1,1,-1).to(boxes.device)
        # if self.add_points:
        #     point_null =  self.null_point_feature.view(1,1,-1).to(boxes.device)
        # if self.add_scribbles:
        #     scribble_null =  self.null_scribble_feature.view(1,1,-1).to(scribbles.device)
        # if self.add_masks:
        #     polygon_null =  self.null_polygon_feature.view(1,1,-1).to(polygons.device)
        # if self.use_segs:
        #     seg_null = self.null_seg_feature.view(1,1,-1).to(segs.device)
        #     seg_null = seg_null.repeat(B,self.num_tokens,1)
        
        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        if self.use_seperate_tokenizer:
            embeddings_list = []
        if self.add_boxes:
            # replace padding with learnable null embedding for boxes
            xyxy_masks = torch.zeros_like(masks).to(masks.device) if drop_box else masks.detach().clone()
            xyxy_embedding = xyxy_embedding*xyxy_masks + (1-xyxy_masks)*xyxy_null
            if self.use_seperate_tokenizer:
                embeddings_list.append(xyxy_embedding)
        # if self.add_points:
        #     # replace padding with learnable null embedding for points
        #     point_masks = torch.zeros_like(masks).to(boxes.device) if drop_point else masks.detach().clone()
        #     point_embedding = point_embedding*point_masks + (1-point_masks)*point_null
        #     if self.use_seperate_tokenizer:
        #         embeddings_list.append(point_embedding)
        # if self.add_scribbles:
        #     # sum along the batch dimension and check if all scribbles are 0s
        #     # replace padding with learnable null embedding for scribbles
        #     # scribble_embedding: torch.Size([bs, n_objs, 640]); masks_scribble: torch.Size([bs, n_objs, 1]); scribble_null: torch.Size([1, 1, 640])
        #     masks_scribble = torch.zeros_like(masks).to(masks.device) if drop_scribble else ((torch.sum(scribbles, dim=-1).unsqueeze(-1) + masks.detach().clone()) > 0).float()
        #     scribble_embedding = scribble_embedding*masks_scribble + (1-masks_scribble)*scribble_null
        #     if self.use_seperate_tokenizer:
        #         embeddings_list.append(scribble_embedding)
        # if self.add_masks:
        #     masks_polygons = torch.zeros_like(masks).to(masks.device) if drop_polygons else ((torch.sum(polygons, dim=-1).unsqueeze(-1) + masks.detach().clone()) > 0).float()
        #     assert torch.sum(scribbles, dim=-1).unsqueeze(-1).size() == masks.size()
        #     polygon_embedding = polygon_embedding*masks_polygons + (1-masks_polygons)*polygon_null
        #     if self.use_seperate_tokenizer:
        #         embeddings_list.append(polygon_embedding)
        # if self.use_segs:
        #     # mask replacing 
        #     masks_segs = torch.zeros(masks.shape[0]).to(masks.device) if drop_segs else (torch.sum(segs, dim=(1,2,3)) > 0).float()
        #     masks_segs = masks_segs.view(-1,1,1)
        #     assert masks_segs.size()[0] == masks.shape[0]
        #     seg_embedding = segs_feature*masks_segs
        #     seg_embedding = seg_embedding + (1-masks_segs)*seg_null 
        #     # add pos 
        #     seg_embedding = seg_embedding + self.pos_embedding 
        #     if self.use_seperate_tokenizer:
        #         embeddings_list.append(seg_embedding)

        inputs = [positive_embeddings]
        if self.use_seperate_tokenizer:
            objs = []
            # forward all types of embeddings using the corresponding linear layers
            for i, (linears, layout_embeddings) in enumerate(zip(self.linears_list, embeddings_list)):
                if i == len(embeddings_list) - 1 and self.use_segs:
                    objs.append(linears( layout_embeddings ) )
                else:
                    objs.append(linears(torch.cat([positive_embeddings, layout_embeddings], dim=-1)))
            objs = torch.cat(objs, dim=1)
        else:
            # NOTE: bbox-only path
            if self.add_boxes:
                inputs.append(xyxy_embedding)
            # if self.add_points:
            #     inputs.append(point_embedding)
            # if self.add_scribbles:
            #     inputs.append(scribble_embedding)
            # if self.add_masks:
            #     inputs.append(polygon_embedding)
        
            objs = self.linears(  torch.cat(inputs, dim=-1)  )
            assert objs.shape == torch.Size([B,N,self.out_dim])       
        # BBOX-ONLY: no polygons branch; keep attention masking enabled
        drop_box_mask = False
        return objs, drop_box_mask
