import torch
import clip
import os
import torchvision.transforms as T
import argparse
import spacy
import numpy as np
import cv2
from PIL import Image
import math

from HybridGL.model.backbone import CLIPViTFM
from HybridGL.utils import extract_noun_phrase, gen_dir_mask, extract_dir_phrase, extract_rela_word, relation_boxes, extract_nouns

import gem
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Target output size (480x720)
TARGET_HEIGHT, TARGET_WIDTH = 480, 720
# Model input size
Height, Width = 224, 224


def process_frame(frame, Height, Width, ref_text, model, gem_model, nlp, mask_generator, device):
    """Output only the original mask tensor after single frame processing, remove all image generation and return logic"""
    # Frame format conversion and preprocessing
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    preprocess = gem.get_gem_img_transform()
    tensor_img = preprocess(img).unsqueeze(0).to(device)
    sam_img = np.array(img)

    # Image tensor normalization
    img_tensor = T.ToTensor()(img)
    img_tensor = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
    image_height, image_width = np.asarray(img).shape[-2], np.asarray(img).shape[-1]

    # Generate SAM masks and bounding boxes
    original_img = img_tensor.to(device)
    sam_masks = mask_generator.generate(sam_img)
    masks = [torch.tensor(m['segmentation']) for m in sam_masks]
    masks = torch.stack(masks).to(device)
    boxes = torch.tensor([m['bbox'] for m in sam_masks]).to(device)

    # Preprocess global/local image features
    pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(masks.device)
    global_imgs = []
    local_imgs = []
    imagesrc = sam_img.copy()
    blurred = cv2.GaussianBlur(imagesrc.copy(), (15, 15), 0)

    for pred_box, pred_mask in zip(boxes, masks):
        pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
        
        # Global image processing
        mask_np = pred_mask.cpu().numpy()
        sharp_region = cv2.bitwise_and(imagesrc, imagesrc, mask=np.clip(mask_np, 0, 255).astype(np.uint8))
        inv_mask = 1 - mask_np
        blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
        global_img = cv2.add(sharp_region, blurred_region)
        global_img = T.ToTensor()(global_img)
        global_img = T.Resize((Height, Width), antialias=None)(global_img)
        global_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(global_img)
        global_imgs.append(global_img)
        
        # Local image processing
        masked_image = original_img * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
        masked_image = T.Resize((Height, Width), antialias=None)(masked_image.squeeze(0))
        local_imgs.append(masked_image.squeeze(0))

    # Stack image features and calculate hybrid features
    global_imgs = torch.stack(global_imgs, dim=0).to(device)
    local_imgs = torch.stack(local_imgs, dim=0).to(device)
    hybrid_features = model(
        local_imgs=local_imgs, 
        global_imgs=global_imgs, 
        pred_masks=masks, 
        fusion_mode="G2L", 
        masking_block=9
    )

    # Text feature calculation
    sentence = ref_text.lower()
    doc = nlp(sentence)
    sentence_for_spacy = ' '.join([token.text for token in doc if token.text != ' '])
    
    # Extract noun phrases and text features
    noun_phrase, _, _ = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
    sentence_token = clip.tokenize(sentence_for_spacy).to(device)
    noun_phrase_token = clip.tokenize(noun_phrase).to(device)
    
    sentence_features = model.model.encode_text(sentence_token)
    noun_phrase_features = model.model.encode_text(noun_phrase_token)
    text_ensemble = 0.5 * sentence_features + 0.5 * noun_phrase_features

    # Calculate positive and negative sample scores
    score_clip = model.calculate_score(hybrid_features, text_ensemble)
    other_noun_phrases, _ = extract_nouns(sentence_for_spacy, nlp)
    other_noun_features = torch.zeros(1, 512).to(device)
    cnt_other_nouns = 0
    for other_noun in other_noun_phrases:
        noun_token = clip.tokenize('a photo of ' + other_noun).to(device)
        other_noun_features += model.model.encode_text(noun_token)
        cnt_other_nouns += 1
    if cnt_other_nouns != 0:
        other_noun_features /= cnt_other_nouns
    score_clip_Neg = model.calculate_score(hybrid_features, other_noun_features)

    # Score normalization
    softmax0 = torch.nn.Softmax(0).to(device)
    score_clip = softmax0(score_clip)
    score_clip_Neg = softmax0(score_clip_Neg)

    # Spatial relationship guided scoring
    relaflag = extract_rela_word(sentence_for_spacy, nlp)
    k1, k2 = min(3, len(score_clip)), min(6, len(score_clip_Neg))
    _, maxidxs = torch.topk(score_clip.view(-1), k=k1)
    _, maxNegidxs = torch.topk(score_clip_Neg.view(-1), k=k2)

    topscores = np.zeros(k1)
    if len(other_noun_phrases) == 0:
        for idx_i in range(k1):
            for idx_j in maxidxs:
                topscores[idx_i] += relation_boxes(
                    boxes[maxidxs[idx_i]], boxes[idx_j],
                    score_clip[maxidxs[idx_i]][0], score_clip[idx_j][0],
                    relaflag
                )
    else:
        for idx_i in range(k1):
            for idx_j in maxNegidxs:
                topscores[idx_i] += relation_boxes(
                    boxes[maxidxs[idx_i]], boxes[idx_j],
                    score_clip[maxidxs[idx_i]][0], score_clip_Neg[idx_j][0],
                    relaflag
                )
    topscores = softmax0(torch.Tensor(topscores).to(device))

    # Spatial consistency guided scoring
    alpha = 0.6
    imgattn = gem_model(tensor_img, [noun_phrase])[0]
    imgattn = T.Resize((image_height, image_width), antialias=True)(imgattn)[0].to(device)
    
    # Fix imgattn channel dimension
    if imgattn.dim() == 3:
        imgattn = imgattn.squeeze(0) if imgattn.shape[0] == 1 else torch.mean(imgattn, dim=0)
    
    # Normalization and spatial guidance
    imgattn = (imgattn - imgattn.min()) / (imgattn.max() - imgattn.min() + 1e-8)
    dirflag = extract_dir_phrase(sentence_for_spacy, nlp, False)
    pmask = gen_dir_mask(dirflag, imgattn.shape[0], imgattn.shape[1], imgattn.device)
    imgattn = (imgattn * pmask) / (imgattn.mean() + 1e-8)

    # Calculate GEM scores
    black = 1.95 if relaflag == "big" else 1.5 if relaflag == "small" else 1.8
    score_gem_list = []
    for pred_mask in masks:
        pred_mask = pred_mask.type(torch.uint8)
        # Match imgattn shape
        if pred_mask.shape != imgattn.shape:
            pred_mask = T.Resize(imgattn.shape, antialias=None)(pred_mask.unsqueeze(0)).squeeze(0)
            pred_mask = (pred_mask > 0.5).float()
        
        # Score calculation
        mask_sum = pred_mask.sum() + 1e-8
        inv_mask_sum = (1 - pred_mask).sum() + 1e-8
        score_gemtmp = (imgattn * (2 - black) * pred_mask / mask_sum).sum() - \
                      (imgattn * black * (1 - pred_mask) / inv_mask_sum).sum()
        score_gem_list.append(torch.Tensor([score_gemtmp]))
    score_gem = torch.stack(score_gem_list, dim=0).to(device)

    # Comprehensive scoring and final mask selection
    for idx_i in range(k1):
        topscores[idx_i] = topscores[idx_i] * (1 - alpha) + alpha * score_gem[maxidxs[idx_i]][0]
    max_index_final = maxidxs[torch.argmax(topscores)]
    
    # Return only the original mask tensor (no images)
    return masks[max_index_final]


def process_video(args):
    """Process video and save only the final mask tensor, remove all other file saving logic"""
    # Device initialization
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model initialization (only necessary models)
    gem_model = gem.create_gem_model(model_name='ViT-B/16', pretrained='openai', device=device)
    model = CLIPViTFM(model_name='ViT-B/16').to(device)
    model.eval()
    nlp = spacy.load('en_core_web_lg')

    # SAM model initialization
    sam = sam_model_registry['default'](checkpoint="./HybridGL/checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=800,
    )

    # Open video file
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Failed to open video file: {args.input_video}")
        return

    # Store raw masks for all frames (only necessary variables)
    all_raw_masks = []
    frame_count = 0

    with torch.no_grad():  # Disable gradient calculation to save memory
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            print(f"Processing frame {frame_count}...")
            # Get only single frame mask tensor, no image processing/saving
            raw_mask = process_frame(
                frame, Height, Width, args.ref_text,
                model, gem_model, nlp, mask_generator, device
            )
            all_raw_masks.append(raw_mask)
            frame_count += 1

    # Release video resources
    cap.release()
    print(f"Video processing completed, total frames processed: {frame_count}")

    # Generate final mask tensor (only this step saves files)
    if frame_count > 0:
        # 1. Resize all masks to target size (480x720) and convert to bool type
        resize_transform = T.Resize((TARGET_HEIGHT, TARGET_WIDTH), antialias=None)
        processed_masks = []
        for mask in all_raw_masks:
            resized_mask = resize_transform(mask.unsqueeze(0)).squeeze(0)
            processed_masks.append(resized_mask.bool())

        # 2. Supplement frames to 4k+1 (duplicate last frame)
        k = math.ceil((frame_count - 1) / 4)
        target_frames = 4 * k + 1
        frames_to_add = target_frames - frame_count
        if frames_to_add > 0:
            last_mask = processed_masks[-1]
            processed_masks.extend([last_mask.clone() for _ in range(frames_to_add)])
        

        # 3. Concatenate into final tensor and save (only save this file)
        final_tensor = torch.stack(processed_masks, dim=0)
        torch.save(final_tensor, args.output_path)
        print(f"Final mask tensor saved to: {args.output_path}")
        print(f"Tensor shape: {final_tensor.shape}")  # Shape should be [4k+1, 480, 720]
    else:
        print("No frames processed, cannot generate mask tensor")


if __name__ == "__main__":
    # Command line argument parsing (only necessary parameters)
    parser = argparse.ArgumentParser(description='Generate and save only video mask tensor')
    parser.add_argument('--input_video', required=True, help='Input video path')
    parser.add_argument('--output_path', required=True, help='Final mask tensor save path (e.g., ./mask_tensor.pt)')
    parser.add_argument('--ref_text', required=True, help='Segmentation reference text (e.g., "the bear")')
    parser.add_argument('--device', help='Computation device (optional, e.g., cuda/cpu, auto-detected by default)')
    
    args = parser.parse_args()
    process_video(args)
