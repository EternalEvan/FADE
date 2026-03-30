model_path="zai-org/CogVideoX-5b"
init_prompt="A brown bear walking slowly along a rocky enclosure in a zoo-like setting. The bear's thick fur has a rich, earthy tone that contrasts against the stone walls and rugged rocks surrounding it. Each frame captures a slight change in the bear's posture as it moves forward, suggesting a calm, natural pace. Sunlight filters through, casting soft shadows and highlighting the textures of the bear's fur and the surrounding rocks. The scene has a serene, almost contemplative quality, focusing on the bear's graceful movements and the peaceful atmosphere of its enclosure."
edit_prompt="A panda walking slowly along a rocky enclosure in a zoo-like setting. The panda's black-and-white fur stands out against the stone walls and rugged rocks around it, creating a striking contrast. Each frame shows a subtle shift in the panda's posture as it moves forward at a calm, deliberate pace. Sunlight filters through, casting gentle shadows and illuminating the textures of the panda's fur and the surrounding rocks. The scene feels serene and tranquil, emphasizing the panda's gentle movements and the peaceful atmosphere of its naturalistic enclosure."
input_video_path="input/bear.mp4"
latent_trajectory_save_path="inversion/bear.pt"
mask_save_path="mask/bear.pt"
output_path="output/bear.mp4"
device="cuda:0"
fps=4
ref_text="a bear"

python ddim_inversion.py \
       --model_path "$model_path" \
       --prompt "$init_prompt" \
       --video_path "$input_video_path" \
       --output_path "$latent_trajectory_save_path" \
       --fps "$fps" \
       --device "$device" 

python generate_mask.py \
    --input_video "$input_video_path" \
    --output_path "$mask_save_path" \
    --ref_text "$ref_text" \
    --device "$device"


python edit.py \
       --model_path "$model_path" \
       --init_prompt "$init_prompt" \
       --edit_prompt "$edit_prompt" \
       --device "$device" \
       --dtype "bf16" \
       --input_video_path "$input_video_path" \
       --latent_trajectory_path "$latent_trajectory_save_path" \
       --mask_path "$mask_save_path" \
       --config_path "configs/bear.yaml" \
       --fps "$fps" \
       --output_video_path "$output_path"