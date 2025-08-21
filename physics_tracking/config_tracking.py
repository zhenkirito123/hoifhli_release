env_cls = "TrackingHumanoid"
# env_cls = "ICCGANHumanoidDemo"

env_params = dict(
    fps = 30,
    random_init = True,
    episode_length = 500,
    character_model = ["assets/humanoid.xml"], # first character must be the humanoid character

    contactable_links = dict( # link_name: contact_height_threshold (default: 0.2)
        L_Ankle = -1000, R_Ankle = -1000,   # allow to contact always
        L_Toe_tip = -1000, R_Toe_tip = -1000,
    ),
    key_link_weights_orient = dict(
        Pelvis = 1.0, 
        L_Hip = 0.5, L_Knee = 0.3, L_Ankle = 0.2, #L_Toe = 0.1,
        R_Hip = 0.5, R_Knee = 0.3, R_Ankle = 0.2, #R_Toe = 0.1,
        Torso = 0.2, Spine = 0.2, Chest = 0.2, Neck = 0.2, Head = 0.2,
        L_Thorax = 0.1, L_Shoulder = 0.2, L_Elbow = 0.2, L_Wrist = 0.3,
        L_Index1 = 0, L_Index2 = 0, L_Index3 = 0, L_Index4 = 0,
        L_Middle1 = 0, L_Middle2 = 0, L_Middle3 = 0, L_Middle4 = 0,
        L_Pinky1 = 0, L_Pinky2 = 0, L_Pinky3 = 0, L_Pinky4 = 0,
        L_Ring1 = 0, L_Ring2 = 0, L_Ring3 = 0, L_Ring4 = 0,
        L_Thumb1 = 0, L_Thumb2 = 0, L_Thumb3 = 0, L_Thumb4 = 0,
        R_Thorax = 0.1, R_Shoulder = 0.2, R_Elbow = 0.2, R_Wrist = 0.3,
        R_Index1 = 0, R_Index2 = 0, R_Index3 = 0, R_Index4 = 0,
        R_Middle1 = 0, R_Middle2 = 0, R_Middle3 = 0, R_Middle4 = 0,
        R_Pinky1 = 0, R_Pinky2 = 0, R_Pinky3 = 0, R_Pinky4 = 0,
        R_Ring1 = 0, R_Ring2 = 0, R_Ring3 = 0, R_Ring4 = 0,
        R_Thumb1 = 0, R_Thumb2 = 0, R_Thumb3 = 0, R_Thumb4 = 0,
        object = 1
    ),
    key_link_weights_pos = dict(
        Pelvis = 1,
        L_Hip = 0, L_Knee = 0, L_Ankle = 0.1, #L_Toe = 0,
        R_Hip = 0, R_Knee = 0, R_Ankle = 0.1, #R_Toe = 0,
        Torso = 0, Spine = 0, Chest = 0, Neck = 0, Head = 0,
        L_Thorax = 0, L_Shoulder = 0, L_Elbow = 0, L_Wrist = 0.3,
        L_Index1 = 0, L_Index2 = 0, L_Index3 = 0, L_Index4 = 0,
        L_Middle1 = 0, L_Middle2 = 0, L_Middle3 = 0, L_Middle4 = 0,
        L_Pinky1 = 0, L_Pinky2 = 0, L_Pinky3 = 0, L_Pinky4 = 0,
        L_Ring1 = 0, L_Ring2 = 0, L_Ring3 = 0, L_Ring4 = 0,
        L_Thumb1 = 0, L_Thumb2 = 0, L_Thumb3 = 0, L_Thumb4 = 0,
        R_Thorax = 0, R_Shoulder = 0, R_Elbow = 0, R_Wrist = 0.3,
        R_Index1 = 0, R_Index2 = 0, R_Index3 = 0, R_Index4 = 0,
        R_Middle1 = 0, R_Middle2 = 0, R_Middle3 = 0, R_Middle4 = 0,
        R_Pinky1 = 0, R_Pinky2 = 0, R_Pinky3 = 0, R_Pinky4 = 0,
        R_Ring1 = 0, R_Ring2 = 0, R_Ring3 = 0, R_Ring4 = 0,
        R_Thumb1 = 0, R_Thumb2 = 0, R_Thumb3 = 0, R_Thumb4 = 0,
        object =1
    ),
    key_link_weights_pos_related = dict(
        Pelvis = 0,
        L_Hip = 0, L_Knee = 0, L_Ankle = 0, #L_Toe = 0,
        R_Hip = 0, R_Knee = 0, R_Ankle = 0, # R_Toe = 0,
        Torso = 0, Spine = 0, Chest = 0, Neck = 0, Head = 0,
        L_Thorax = 0, L_Shoulder = 0, L_Elbow = 0, L_Wrist = 1,
        L_Index1= .3, L_Index2= .3, L_Index3= .3, L_Index4= .3,
        L_Middle1= .3, L_Middle2= .3, L_Middle3= .3, L_Middle4= .3,
        L_Pinky1= .3, L_Pinky2= .3, L_Pinky3= .3, L_Pinky4= .3,
        L_Ring1= .3, L_Ring2= .3, L_Ring3= .3, L_Ring4= .3,
        L_Thumb1= .3, L_Thumb2= .3, L_Thumb3= .3, L_Thumb4= .3,
        R_Thorax = 0, R_Shoulder = 0, R_Elbow = 0, R_Wrist = 1,
        R_Index1= .3, R_Index2= .3, R_Index3= .3, R_Index4= .3,
        R_Middle1= .3, R_Middle2= .3, R_Middle3= .3, R_Middle4= .3,
        R_Pinky1= .3, R_Pinky2= .3, R_Pinky3= .3, R_Pinky4= .3,
        R_Ring1= .3, R_Ring2= .3, R_Ring3= .3, R_Ring4= .3,
        R_Thumb1= .3, R_Thumb2= .3, R_Thumb3= .3, R_Thumb4= .3
    ),
    key_link_weights_acc_penalty = dict(
        L_Ankle=1, L_Toe_tip=1,
        R_Ankle=1, R_Toe_tip=1,
        L_Wrist=1, R_Wrist=1
    )
)

training_params = dict(
    max_epochs =  1000000,
    save_interval = 50000
)

discriminators = {}
