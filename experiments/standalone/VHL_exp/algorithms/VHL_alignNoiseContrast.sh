
export VHL=${VHL:-True}
export VHL_save_images=${VHL_save_images:-False}
export VHL_data=${VHL_data:-dataset}
export VHL_dataset_list=${VHL_dataset_list:-"['style_GAN_init']"}

export VHL_label_from=${VHL_label_from:-dataset}
export VHL_label_style=${VHL_label_style:-extra}

export VHL_generator=${VHL_generator:-style_GAN_v2_G}
export VHL_generator_from_server=${VHL_generator_from_server:-True}
export VHL_num=${VHL_num:-10}
export VHL_generator_num=${VHL_generator_num:-1}
export VHL_alpha=${VHL_alpha:-1.0}

export image_resolution=${image_resolution:-32}
export style_gan_style_dim=${style_gan_style_dim:-64}
export style_gan_n_mlp=${style_gan_n_mlp:-1}
export style_gan_cmul=${style_gan_cmul:-1}
export style_gan_sample_z_mean=${style_gan_sample_z_mean:-0.3}
export style_gan_sample_z_std=${style_gan_sample_z_std:-0.3}


export model_out_feature=True
export VHL_feat_align=True
export VHL_feat_align_inter_domain_weight=0.0
export VHL_feat_align_inter_cls_weight=${VHL_feat_align_inter_cls_weight:-1.0}
export VHL_noise_supcon_weight=${VHL_noise_supcon_weight:-0.1}
export model_out_feature_layer=${model_out_feature_layer:-last}

export VHL_inter_domain_mapping=False
export VHL_inter_domain_ortho_mapping=False
export VHL_class_match=True
export VHL_feat_detach=True
export VHL_noise_contrastive=True
export VHL_data_re_norm=False
export VHL_shift_test=True