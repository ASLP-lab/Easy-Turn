import argparse
import io
import logging
import os
import tarfile
import time
import multiprocessing
import json

import torch
import torchaudio
from tqdm import tqdm  # 需要安装 tqdm 库

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def write_tar_file(data_list, tar_file, resample=16000, index=0, total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0

    with tarfile.open(tar_file, "w") as tar:
        # 使用 tqdm 显示进度条
        for i, item in enumerate(tqdm(data_list, desc=f'Creating {os.path.basename(tar_file)}', unit='file')):
            #设置要压缩的项
            task = item["task"]
            key = item["key"]
            txt = item["txt"]
            wav = item["wav"]
            lang = item["lang"]
            #duration = item["duration"]
            speaker = item["speaker"]
            emotion = item["emotion"]
            gender = item["gender"]
            state = item["state"]
            extra = item["extra"]

            # logging.info(f"Processing audio file: {wav}")

            suffix = wav.split('.')[-1]
            if suffix not in AUDIO_FORMAT_SETS:
                logging.warning(f"File format {suffix} not supported. Skipping file {wav}")
                continue

            try:
                # Process audio
                ts = time.time()
                audio, sample_rate = torchaudio.load(wav)
                audio = torchaudio.transforms.Resample(sample_rate, resample)(audio)
                read_time += (time.time() - ts)

                audio = (audio * (1 << 15)).to(torch.int16)
                ts = time.time()
                with io.BytesIO() as f:
                    torchaudio.save(f, audio, resample, format="wav", bits_per_sample=16)
                    suffix = "wav"
                    f.seek(0)
                    data = f.read()
                save_time += (time.time() - ts)

                ts = time.time()
                # Save text file
                txt_file = key + '.txt'
                txt = txt.encode('utf8')
                txt_data = io.BytesIO(txt)
                txt_info = tarfile.TarInfo(txt_file)
                txt_info.size = len(txt)
                tar.addfile(txt_info, txt_data)

                # Save wav file
                wav_file = key + '.' + suffix
                wav_data = io.BytesIO(data)
                wav_info = tarfile.TarInfo(wav_file)
                wav_info.size = len(data)
                tar.addfile(wav_info, wav_data)

                # Save metadata fields (task, lang, speaker, emotion, gender, duration....) each in separate files
                for field, value in {"task": task, "lang": lang, "speaker": speaker, "emotion": emotion, "gender": gender, "state": state}.items():
                    field_file = f"{key}.{field}"  # 文件名格式修改
                    field_data = io.BytesIO(str(value).encode('utf8'))
                    field_info = tarfile.TarInfo(field_file)
                    field_info.size = len(str(value))
                    tar.addfile(field_info, field_data)
                    # logging.info(f"Added file {field_file} with content: {value}")

                # Extract duration from extra and save it to a separate file
                
                duration = 0
                duration_file = key + '.duration'
                duration_data = io.BytesIO(str(duration).encode('utf8'))
                duration_info = tarfile.TarInfo(duration_file)
                duration_info.size = len(str(duration))
                tar.addfile(duration_info, duration_data)
                

                # Save remaining extra data (excluding duration) to a separate key.extra file
                remaining_extra = {k: v for k, v in extra.items() if k != "duration"}
                jsonl_line = json.dumps(remaining_extra, ensure_ascii=False)
                jsonl_data = jsonl_line.encode('utf8')
                jsonl_file = key + '.extra'  # 文件名格式修改
                jsonl_data_io = io.BytesIO(jsonl_data)
                jsonl_info = tarfile.TarInfo(jsonl_file)
                jsonl_info.size = len(jsonl_data)
                tar.addfile(jsonl_info, jsonl_data_io)

                write_time += (time.time() - ts)
            except Exception as e:
                logging.error(f"Error processing file {wav}: {e}")

        logging.info(f'read {read_time:.2f}s save {save_time:.2f}s write {write_time:.2f}s')

if __name__ == '__main__':
    print("Welcome!")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--resample',
                        type=int,
                        default=16000,
                        help='resample rate for audio files')
    parser.add_argument('data_file', help='input JSON list file')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)

    data = []
    with open(args.data_file, 'r', encoding='utf8') as fin:
        for line in fin:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON on line: {line.strip()}")
                raise e

    logging.info(f"Total records loaded: {len(data)}")

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # 使用线程池加速处理
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir, '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(
            write_tar_file,
            (chunk, tar_file, args.resample, i, num_chunks))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
