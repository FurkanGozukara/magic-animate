# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import imageio
import numpy as np
import gradio as gr
from PIL import Image
import subprocess
import os

from demo.animate import MagicAnimate

animator = MagicAnimate()

def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
    return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale)

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/magic-research/magic-animate" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                <a href="https://arxiv.org/abs/2311.16498"><img src="https://img.shields.io/badge/Arxiv-2311.16498-red"></a>
                <a href='https://showlab.github.io/magicanimate'><img src='https://img.shields.io/badge/Project_Page-MagicAnimate-green' alt='Project Page'></a>
                <a href='https://github.com/magic-research/magic-animate'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            </div>
        </div>
        </div>
        """
    )
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
    
    with gr.Row():
        reference_image  = gr.Image(label="Reference Image")
        motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video):
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        
        if fps != 25.0:
            # Define the name for the converted video
            converted_video = "converted_video.mp4"

            # Use FFmpeg to convert the video to 25 fps
            subprocess.run(['ffmpeg', '-i', video, '-r', '25', converted_video], check=True)

            # Check if the conversion was successful and the file exists
            if os.path.exists(converted_video):
                return converted_video
            else:
                raise Exception("Video conversion failed.")
        else:
            # If the video is already 25 fps, return it as is
            return video
    
    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))
    
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"]
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

demo.launch(share=False,inbrowser=True)
