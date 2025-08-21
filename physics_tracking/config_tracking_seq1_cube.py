from config_tracking import *
# env_cls = "ICCGANHumanoidDemo"
env_params["character_model"] += ["assets/largebox_cube.urdf", "assets/largebox_1_cube.urdf", "assets/largebox_2_cube.urdf", "assets/largebox_3_cube.urdf"]
env_params["motion_file"] = "seq_1.json"
env_params["contactable_links"]["largebox"] = -1000
env_params["contactable_links"]["largebox_1"] = -1000
env_params["contactable_links"]["largebox_2"] = -1000
env_params["contactable_links"]["largebox_3"] = -1000
for s in ["L_", "R_"]:
    env_params["contactable_links"][s+"Wrist"] = 0.1
    for i in range(1, 5):
        for f in ["Index", "Middle", "Ring", "Pinky", "Thumb"]:
            env_params["contactable_links"][s+f+str(i)] = 0.1
