import os
import subprocess

import imageio
import numpy as np
import trimesh


def concat_multiple_videos(input_files, output_file):
    # List of input files
    # input_files = ['video1.mp4', 'video2.mp4']

    # Output file
    # output_file = 'output.mp4'

    # Step 1: Convert each video to a consistent FPS (e.g., 30 fps) and save to a temp file.
    temp_files = []
    target_fps = 30

    temp_folder = output_file.replace(".mp4", "_tmp")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    for i, file in enumerate(input_files):
        temp_filename = os.path.join(temp_folder, str(i) + ".mp4")

        temp_files.append(temp_filename)
        reader = imageio.get_reader(file)
        writer = imageio.get_writer(temp_filename, fps=target_fps)

        for frame in reader:
            writer.append_data(frame)
        writer.close()

    # Step 2: Concatenate the temporary files.
    with imageio.get_writer(output_file, fps=target_fps) as final_writer:
        for temp_file in temp_files:
            reader = imageio.get_reader(temp_file)
            for frame in reader:
                final_writer.append_data(frame)

    # Step 3: Cleanup temp files.
    # for temp_file in temp_files:
    #     os.remove(temp_file)
    # #     shutil.rmtree(temp_folder)
    # shutil.rmtree(temp_folder)


def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        "ffmpeg",
        "-r",
        "30",
        "-y",
        "-threads",
        "16",
        "-i",
        f"{img_folder}/%05d.png",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def images_to_video_w_imageio(img_folder, output_vid_file, fps=30):
    img_files = os.listdir(img_folder)
    img_files.sort()
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=fps, quality=8)


def save_verts_faces_to_mesh_file(
    mesh_verts, mesh_faces, save_mesh_folder, save_gt=False
):
    # mesh_verts: T X Nv X 3
    # mesh_faces: Nf X 3
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx], faces=mesh_faces)
        if save_gt:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d" % (idx) + "_gt.ply")
        else:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d" % (idx) + ".ply")
        mesh.export(curr_mesh_path)


def save_verts_faces_to_mesh_file_w_object(
    mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder
):
    # mesh_verts: T X Nv X 3
    # mesh_faces: Nf X 3
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx], faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d" % (idx) + ".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
        curr_obj_mesh_path = os.path.join(
            save_mesh_folder, "%05d" % (idx) + "_object.ply"
        )
        obj_mesh.export(curr_obj_mesh_path)
