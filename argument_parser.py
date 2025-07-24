import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vis_wdir", type=str, default="default_folder")

    parser.add_argument("--navi_save_dir", type=str, default="navi_release")
    parser.add_argument("--rnet_save_dir", type=str, default="rnet_release")
    parser.add_argument("--cnet_save_dir", type=str, default="cnet_release")

    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--wandb_pj_name", type=str, default="", help="project name")
    parser.add_argument("--entity", default="zhenkirito123", help="W&B entity")
    parser.add_argument("--exp_name", default="", help="save to project/name")
    parser.add_argument("--device", default="0", help="cuda device")

    parser.add_argument("--window", type=int, default=120, help="horizon")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="generator_learning_rate"
    )

    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint")

    parser.add_argument("--data_root_folder", type=str, default="", help="checkpoint")

    parser.add_argument(
        "--n_dec_layers_nav", type=int, default=4, help="the number of decoder layers"
    )
    parser.add_argument(
        "--n_head_nav",
        type=int,
        default=4,
        help="the number of heads in self-attention",
    )
    parser.add_argument(
        "--d_k_nav", type=int, default=256, help="the dimension of keys in transformer"
    )
    parser.add_argument(
        "--d_v_nav",
        type=int,
        default=256,
        help="the dimension of values in transformer",
    )
    parser.add_argument(
        "--d_model_nav",
        type=int,
        default=512,
        help="the dimension of intermediate representation in transformer",
    )

    # For testing sampled results
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset
    parser.add_argument("--test_on_train", action="store_true")

    # For loss type
    parser.add_argument("--use_l2_loss", action="store_true")

    parser.add_argument("--compute_metrics", action="store_true")

    # For interaction model setting
    # For adding full body prediction.
    parser.add_argument(
        "--n_dec_layers", type=int, default=4, help="the number of decoder layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=4, help="the number of heads in self-attention"
    )
    parser.add_argument(
        "--d_k", type=int, default=256, help="the dimension of keys in transformer"
    )
    parser.add_argument(
        "--d_v", type=int, default=256, help="the dimension of values in transformer"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="the dimension of intermediate representation in transformer",
    )

    # Add language conditions.
    parser.add_argument("--add_contact_label", action="store_true")

    # Train and test on different objects.
    parser.add_argument("--use_object_split", action="store_true")

    # For adding start and end object position (xyz) and rotation (6D rotation).
    parser.add_argument("--add_start_end_object_pos_rot", action="store_true")

    # For adding start and end object position (xyz).
    parser.add_argument("--add_start_end_object_pos", action="store_true")

    # For adding start and end object position at z plane (xy).
    parser.add_argument("--add_start_end_object_pos_xy", action="store_true")

    # Random sample waypoints instead of fixed intervals.
    parser.add_argument("--use_random_waypoints", action="store_true")

    # Input the first human pose, maybe can connect the windows better.
    parser.add_argument("--remove_target_z", action="store_true")

    # Input the first human pose, maybe can connect the windows better.
    parser.add_argument("--use_guidance_in_denoising", action="store_true")

    parser.add_argument("--use_optimization_in_denoising", action="store_true")

    # Add rest offsets for body shape information.
    parser.add_argument("--add_rest_human_skeleton", action="store_true")

    # Add rest offsets for body shape information.
    parser.add_argument("--use_first_frame_bps", action="store_true")

    parser.add_argument("--input_full_human_pose", action="store_true")

    parser.add_argument("--use_two_stage_pipeline", action="store_true")

    parser.add_argument("--use_unified_interaction_model", action="store_true")

    # Visualize the results from different noise levels.
    parser.add_argument("--return_diff_level_res", action="store_true")

    parser.add_argument(
        "--test_object_names",
        type=str,
        nargs="+",
        default=["largebox"],
        help="object names for long sequence generation testing",
    )

    parser.add_argument("--use_long_planned_path", action="store_true")

    parser.add_argument("--use_long_planned_path_only_navi", action="store_true")

    # For testing sampled results w planned path
    parser.add_argument("--use_planned_path", action="store_true")

    parser.add_argument(
        "--loss_w_feet",
        type=float,
        default=1,
        help="the loss weight for feet contact loss",
    )

    parser.add_argument(
        "--loss_w_fk", type=float, default=1, help="the loss weight for fk loss"
    )

    parser.add_argument(
        "--loss_w_obj_pts", type=float, default=1, help="the loss weight for fk loss"
    )

    # For interaction model setting.
    parser.add_argument("--add_interaction_root_xy_ori", action="store_true",)
    parser.add_argument("--add_interaction_feet_contact", action="store_true",)
    parser.add_argument("--add_wrist_relative", action="store_true")
    parser.add_argument("--add_object_static", action="store_true")

    parser.add_argument("--use_noisy_traj", action="store_true")

    ############################ finger motion related settings ############################
    parser.add_argument("--add_finger_motion", action="store_true")

    parser.add_argument(
        "--finger_project",
        default="./experiments",
        help="output folder for weights and visualizations",
    )
    parser.add_argument(
        "--finger_wandb_pj_name",
        type=str,
        default="omomo_only_finger",
        help="wandb_proj_name",
    )
    parser.add_argument("--finger_entity", default="zhenkirito123", help="W&B entity")
    parser.add_argument(
        "--finger_exp_name",
        default="fnet_release",
        help="save to project/name",
    )
    parser.add_argument("--finger_device", default="0", help="cuda device")

    parser.add_argument("--finger_window", type=int, default=30, help="horizon")

    parser.add_argument("--finger_batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--finger_learning_rate",
        type=float,
        default=2e-4,
        help="generator_learning_rate",
    )

    parser.add_argument("--finger_checkpoint", type=str, default="", help="checkpoint")

    parser.add_argument(
        "--finger_n_dec_layers",
        type=int,
        default=4,
        help="the number of decoder layers",
    )
    parser.add_argument(
        "--finger_n_head",
        type=int,
        default=4,
        help="the number of heads in self-attention",
    )
    parser.add_argument(
        "--finger_d_k",
        type=int,
        default=256,
        help="the dimension of keys in transformer",
    )
    parser.add_argument(
        "--finger_d_v",
        type=int,
        default=256,
        help="the dimension of values in transformer",
    )
    parser.add_argument(
        "--finger_d_model",
        type=int,
        default=512,
        help="the dimension of intermediate representation in transformer",
    )

    parser.add_argument("--finger_generate_refine_dataset", action="store_true")
    parser.add_argument("--finger_generate_quant_eval", action="store_true")
    parser.add_argument("--finger_generate_vis_eval", action="store_true")
    parser.add_argument("--finger_milestone", type=str, default="17")
    # For testing sampled results
    parser.add_argument("--finger_test_sample_res", action="store_true")

    # For testing sampled results on training dataset
    parser.add_argument("--finger_test_sample_res_on_train", action="store_true")

    # For loss type
    parser.add_argument("--finger_use_l2_loss", action="store_true")

    # For training diffusion model without condition
    parser.add_argument("--finger_remove_condition", action="store_true")

    # For normalizing condition
    parser.add_argument("--finger_normalize_condition", action="store_true")

    # For adding first human pose as condition (shared by current trainer and FullBody trainer)
    parser.add_argument("--finger_add_start_human_pose", action="store_true")

    # FullBody trainer config: For adding hand pose (translation+rotation) as condition
    parser.add_argument("--finger_add_hand_pose", action="store_true")

    # FullBody trainer config: For adding hand pose (translation only) as condition
    parser.add_argument("--finger_add_hand_trans_only", action="store_true")

    # FullBody trainer config: For adding hand and foot trans as condition
    parser.add_argument("--finger_add_hand_foot_trans", action="store_true")

    # For canonicalizing the first pose's facing direction
    parser.add_argument("--finger_cano_init_pose", action="store_true")

    # For predicting hand position only
    parser.add_argument("--finger_pred_hand_jpos_only_from_obj", action="store_true")

    # For predicting hand position only
    parser.add_argument("--finger_pred_palm_jpos_from_obj", action="store_true")

    # For predicting hand position only
    parser.add_argument("--finger_pred_hand_jpos_and_rot_from_obj", action="store_true")

    # For running the whole pipeline.
    parser.add_argument("--finger_run_whole_pipeline", action="store_true")

    parser.add_argument("--finger_add_object_bps", action="store_true")

    parser.add_argument("--finger_add_hand_processing", action="store_true")

    parser.add_argument("--finger_for_quant_eval", action="store_true")

    parser.add_argument("--finger_use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--finger_use_object_split", action="store_true")

    parser.add_argument("--finger_add_palm_jpos_only", action="store_true")

    parser.add_argument("--finger_use_blender_data", action="store_true")

    parser.add_argument("--finger_use_wandb", action="store_true")

    parser.add_argument("--finger_use_arctic", action="store_true")
    parser.add_argument("--finger_omomo_obj", type=str, default="")

    parser.add_argument("--finger_train_ambient_sensor", action="store_true")
    parser.add_argument("--finger_train_proximity_sensor", action="store_true")
    # parser.add_argument("--finger_train_contact_label", action="store_true")

    parser.add_argument("--finger_wrist_obj_traj_condition", action="store_true")
    parser.add_argument("--finger_ambient_sensor_condition", action="store_true")
    parser.add_argument("--finger_contact_label_condition", action="store_true")
    
    # Set a few args to True by default.
    parser.set_defaults(
        finger_use_joints24=True,
        finger_use_omomo=True,
        finger_train_both_sensor=True,
        finger_proximity_sensor_condition=True,
        finger_ref_pose_condition=True,
    )

    opt = parser.parse_args()
    return opt
