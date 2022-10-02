import tensorflow as tf
from tensorflow import keras
from keras.layers import LayerNormalization, Dropout, GlobalAveragePooling1D, Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras import Model
from .transformer import  SwinTransformerBlock, PatchEmbed, BasicLayer, PatchMerging, Patch_expanding
import numpy as np


class SM_Transformer(Model):
    def __init__(
        self, 
        n_classes,
        name = ''):

        super().__init__()
        self.encoder = SwinUnetEncoder()
        self.decoder = SwinUnetDecoder()

        self.class_proj = Conv2D(filters = n_classes, kernel_size = 1)
    
    
    def call(self, inputs):
        input_0 = inputs[0]
        input_1 = inputs[1]
        previous_input = inputs[2]

        input = tf.concat([input_0,  input_1, previous_input], axis=-1)
        #input = tf.concat([input_0,  input_1], axis=-1)

        x = self.encoder(input)
        x = self.decoder(x)

        #x = self.class_proj(x)
        #x = self.last_bn(x)
        #x = tf.keras.activations.relu(x)

        x = self.class_proj(x)

        return tf.keras.activations.softmax(x)
        #return tf.keras.activations.sigmoid(x)

class SwinUnetEncoder(tf.keras.layers.Layer):
    def __init__(self, model_name='encoder', include_top=False,
                 img_size=(128, 128), patch_size=(4, 4), in_chans=27, num_classes=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 16],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        '''self.basic_layers = [BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)]'''

        self.basic_layer_0 = BasicLayer(dim=int(embed_dim),
                                                input_resolution=(patches_resolution[0],
                                                                  patches_resolution[1]),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:0]):sum(
                                                    depths[:0 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_0')
        self.patch_merging_0 = PatchMerging(
            dim = int(embed_dim),
            input_resolution=(  patches_resolution[0],
                                patches_resolution[1]),
            norm_layer=norm_layer
        )

        self.basic_layer_1 = BasicLayer(dim=int(embed_dim*2),
                                                input_resolution=(patches_resolution[0]//2,
                                                                  patches_resolution[1]//2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:1]):sum(
                                                    depths[:1 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_1')
        self.patch_merging_1 = PatchMerging(
            dim = int(embed_dim*2),
            input_resolution=(  patches_resolution[0]//2,
                                patches_resolution[1]//2),
            norm_layer=norm_layer
        )

        self.basic_layer_2 = BasicLayer(dim=int(embed_dim*4),
                                                input_resolution=(patches_resolution[0]//4,
                                                                  patches_resolution[1]//4),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:2]):sum(
                                                    depths[:2 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_2')
        self.patch_merging_2 = PatchMerging(
            dim = int(embed_dim*4),
            input_resolution=(  patches_resolution[0]//4,
                                patches_resolution[1]//4),
            norm_layer=norm_layer
        )

        self.basic_layer_3 = BasicLayer(dim=int(embed_dim*8),
                                                input_resolution=(patches_resolution[0]//8,
                                                                  patches_resolution[1]//8),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:3]):sum(
                                                    depths[:3 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_3')

        '''self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None
'''
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_1 = self.basic_layer_0(x)
        x_1r = self.patch_merging_0(x_1)

        x_2 = self.basic_layer_1(x_1r)
        x_2r = self.patch_merging_1(x_2)

        x_3 = self.basic_layer_2(x_2r)
        x_3r = self.patch_merging_2(x_3)

        x_4 = self.basic_layer_3(x_3r)
       
        #x = self.norm(x)
        #x = self.avgpool(x)
        return (x_4, x_3, x_2, x_1)

    def call(self, x):
        x = self.forward_features(x)
        #if self.include_top:
        #    x = self.head(x)
        return x

class SwinUnetDecoder(tf.keras.layers.Layer):
    def __init__(self, model_name='decoder', include_top=False,
                 img_size=(128, 128), patch_size=(4, 4), in_chans=27, num_classes=3,
                 embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers

        self.patch_expand_0 = Patch_expanding(
            num_patch = (patches_resolution[0],
                         patches_resolution[1]), 
            embed_dim = embed_dim*4, 
            upsample_rate = 4
        )

        self.basic_layer_0 = BasicLayer(dim=int(embed_dim),
                                                input_resolution=(patches_resolution[0],
                                                                  patches_resolution[1]),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:0]):sum(
                                                    depths[:0 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_0')
        self.patch_expand_1 = Patch_expanding(
            num_patch = (patches_resolution[0]//2,
                         patches_resolution[1]//2), 
            embed_dim = embed_dim*2, 
            upsample_rate = 2
        )


        self.basic_layer_1 = BasicLayer(dim=int(embed_dim*2),
                                                input_resolution=(patches_resolution[0]//2,
                                                                  patches_resolution[1]//2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:1]):sum(
                                                    depths[:1 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_1')

        self.patch_expand_2 = Patch_expanding(
            num_patch = (patches_resolution[0]//4,
                         patches_resolution[1]//4), 
            embed_dim = embed_dim*4, 
            upsample_rate = 2
        )


        self.basic_layer_2 = BasicLayer(dim=int(embed_dim*4),
                                                input_resolution=(patches_resolution[0]//4,
                                                                  patches_resolution[1]//4),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:2]):sum(
                                                    depths[:2 + 1])],
                                                norm_layer=norm_layer,
                                                downsample=None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers_2')

        self.patch_expand_3 = Patch_expanding(
            num_patch = (patches_resolution[0]//8,
                         patches_resolution[1]//8), 
            embed_dim = embed_dim*8, 
            upsample_rate = 2
        )

        self.proj_3 = Conv2D(filters = embed_dim*4, kernel_size = 1)
        self.proj_2 = Conv2D(filters = embed_dim*2, kernel_size = 1)
        self.proj_1 = Conv2D(filters = embed_dim, kernel_size = 1)

        '''self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None'''

    def forward_features(self, x):
        x_3r = x[0]
        x_3s = x[1]
        x_2s = x[2]
        x_1s = x[3]

        x_3e = self.patch_expand_3(x_3r)
        x_3c = tf.concat([x_3e, x_3s], axis = -1)
        x_3_l = self.proj_3(x_3c)
        
        x_2 = self.basic_layer_2(x_3_l)
        x_2e = self.patch_expand_2(x_2)
        x_2c = tf.concat([x_2e, x_2s], axis = -1)
        x_2_l = self.proj_2(x_2c)

        x_1 = self.basic_layer_1(x_2_l)
        x_1e = self.patch_expand_1(x_1)
        x_1c = tf.concat([x_1e, x_1s], axis = -1)
        x_1_l = self.proj_1(x_1c)

        x_0 = self.basic_layer_0(x_1_l)
        x_0 = self.patch_expand_0(x_0)

        return x_0


    def call(self, x):
        x = self.forward_features(x)
        return x

class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, depth):
        super().__init__()

        self.fc1 = Dense(depth // self.ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)

        self.fc2 = Dense(depth, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)

    def call(self, x):
        shortcut = x
        x = GlobalAveragePooling2D(keepdims=True)(x)
        # reduce the number of filters (1 x 1 x C/r)
        x = self.fc1(x)

        # the excitation operation restores the input dimensionality
        x = self.fc2(x)

        # multiply the attention weights with the original input
        x = tf.layers.multiply([shortcut, x])
        # return the output of the SE block
        return x

