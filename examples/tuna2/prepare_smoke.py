"""Create local tiny Qwen weights/tokenizer and four real JSONL data streams."""
import argparse
import json
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/tuna2-smoke')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    llm, data = root / 'tiny-qwen', root / 'data'
    llm.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    tokens = ['[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', '<|vision_start|>',
              '<|vision_end|>', '<|image_pad|>', '<|video_pad|>', '<image>',
              '<|vid_start|>', '<|vid_end|>', 'red', 'blue', 'square', 'make',
              'it', 'Describe', 'the', 'image', '.', 'A', 'is']
    tokenizer = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]',
                                  pad_token='[PAD]', bos_token='<|im_start|>', eos_token='<|im_end|>')
    fast.save_pretrained(llm)
    Qwen2Config(vocab_size=len(tokens), hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=256, use_cache=False).save_pretrained(llm)
    for color in ('red', 'blue'):
        Image.new('RGB', (32, 32), color).save(data / f'{color}.png')
    records = {
        't2i': [{'image': f'{c}.png', 'caption': f'A {c} square.'} for c in ('red', 'blue')],
        'edit': [{'raw_image': 'red.png', 'out_image': 'blue.png', 'instruction': 'make it blue'},
                 {'raw_image': 'blue.png', 'out_image': 'red.png', 'instruction': 'make it red'}],
        'mmu': [{'image': f'{c}.png', 'conversations': [
                    {'from': 'human', 'value': '<image> Describe the image.'},
                    {'from': 'gpt', 'value': f'A {c} square.'}]} for c in ('red', 'blue')],
        'text': [{'caption': 'A red square.'}, {'caption': 'A blue square.'}],
    }
    streams = {}
    for name, rows in records.items():
        file = data / f'{name}.jsonl'
        file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        kw = dict(jsonl_path=str(file), image_root=str(data), image_size=32,
                  max_text_length=64, batch_size=1, num_workers=0, num_image_tokens=4)
        if name == 'edit':
            kw['_target_'] = 'mindspeed_mm.models.tuna2.upstream.data.datasets.edit_dataset.EditDataset'
        else:
            kw['_target_'] = 'mindspeed_mm.models.tuna2.upstream.data.datasets.ti_dataset.TIDataset'
            kw['data_type'] = {'text': 'mmu_text'}.get(name, name)
        streams[name] = kw
    cfg = dict(seed=42, model=dict(llm_model_path=str(llm), init_llm_from_config=True,
               hidden_size=64, image_latent_dim=3, patch_size=16, image_latent_height=2,
               image_latent_width=2, mixed_modal_latent_height=2, mixed_modal_latent_width=2,
               num_attention_heads=4, num_key_value_heads=2, flow_head_num=2,
               add_aspect_ratio_embeds=False, attention_backend='sdpa', dtype='fp32',
               enable_mask_token=True, masked_image_ratio=0.5),
               data=dict(streams=streams, sampling_weights=dict(t2i=.5, edit=.2, mmu=.2, text=.1)),
               training=dict(output_dir=str(root / 'checkpoints'), max_steps=8,
                             learning_rate=2e-4, weight_decay=.01, warmup_steps=2,
                             gradient_accumulation_steps=1, ema_decay=.9999, cycle_streams=True))
    OmegaConf.save(OmegaConf.create(cfg), root / 'smoke.yaml')
    print(root / 'smoke.yaml')


if __name__ == '__main__':
    main()
