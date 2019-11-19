# Tài liệu cho hệ thống TTS
# I. Chuẩn bị dữ liệu
## 1. Chọn giọng đọc
Đưa 2 đoạn văn bản cho các ứng viên đọc, sau đó chọn ra những giọng tốt nhất dựa trên các tiêu chí sau:

-   Giọng đọc hay thu hút
-   Tự nhiên: không cố nhấn nhá
-   Đọc rõ ràng: nghe rõ các từ dù đọc hơi nhanh nhưng không mất chút âm nào.
-   Độ ổn định: phong cách đọc, tốc độ đọc và âm lượng không thay đổi.

## 2. Chuẩn bị transcript đọc
Lấy (crawl) các bài báo trên các trang báo mạng như Vnexpress hay dantri.com.vn đảm bảo:

-   Các lĩnh vực đa dạng: thể thao, sức khỏe, đời sống …    
-   Độ phủ từ ( trên 3k từ: cả từ vay mượn)

## 3. Tổ chức ghi âm
-   Phòng thu: đảm bảo không để tiếng ồn lọt vào, tường có chống vọng và mic có filter.
-   Chiến lược ghi âm: Ban đầu ghi âm 30 phút/ ngày, kiểm duyệt liên tục theo ngày, cho đến khi toàn bộ dữ liệu ngày đó đảm bảo, thì cho họ ghi âm 1h-2h/ ngày.
-   Điểu kiện đảm bảo dữ liệu: độ ổn định giọng đọc, tự nhiên, không vấp, rõ ràng và đặc biệt không có nhiễu trong file audio. Audio ưu tiên dạng định dạng wav.
-   Nghiệm thu dữ liệu hàng ngày: đảm bảo các ngày đọc không khác nhau, và dữ liệu vẫn đảm bảo các tiêu chí nêu trên.

# II. Training tutorial

Phần **Training tutorial** sẽ trình bày chi tiết các bước để có thể huấn luyện được mô hình TTS một cách tối ưu nhất 

## 1. Tacotron2

Cấu trúc project Taocotron2
```
ztacotron2
└───base
	└───data
	│   │   camhieu
	│   │   ...
	└───experiments
	│   │   camhieu
	│   │   ...
	└───dicts
	│   │   phone_oov
	│   │   ...
	└───test_set
	│   │   vlsp2018.txt
	│   │   ...
	│   data_utils.py
	│   decode_time_stamps.py    
	│   distributed.py	
	│   gen_tts.py 
	│   glow.py
	│   hparams.py
	│   layers.py
	│   logger.py
	│   model.py
	│   multiproc.py
	│   oov2phonemes.py
	│   paths.py
	│   prepare_files_list.py
	│   prepare_wavs.py
	│   tacotron2_pretrained.pt
	│   text_embedding.py
	│   train_tacotron_new.py
	│   tts_infer.py
	│   utils.py
	│   vad.py
```

### 1.1. Môi trường training
Môi trường training đã được đóng gói và chuẩn bị sẵn dưới dạng conda enviroment
```
conda activate tacotron
```

### 1.2. Chuẩn bị dữ liệu training
Clone folder **base** để tạo 1 project riêng cho giọng mới
```
ztacotron2
└───base
└───camhieu
```

Thư mục chứa dữ liệu training cho từng giọng được đặt trong thư mục **data**, có cấu trúc như sau:
```
ztacotron2
└───camhieu
	└───data
	    └───camhieu
	        │   metadata.txt
	        └───wavs_raw
	            │   numiennam-camhieu-001-000697-001659.wav
	            │   numiennam-camhieu-001-001755-002716.wav
	            │   ...
```
**wavs_raw** là thư mục chứa các đoạn audio đã được cắt nhỏ ra theo từng câu từ audio gốc, cần đảm bảo các tiêu chí sau:

- Có độ dài từ 2-10s
- Đầu và đuôi audio không được cắt lẹm vào từ
- Các audio có format WAV - PCM 16 bit, sample rate có thể là 16000Hz hoặc 22050Hz tùy vào mô hình huấn luyện

**metadata.txt** là file chứa text của từng đoạn audio ở trên, có format như sau:

```
numiennam-camhieu-001-000697-001659|cho rằng thửa đất nhà mình bị thu hồi , và đền bù không đúng quy định của pháp luật * ông phạm văn phượng ở phố cầu mây , thị trấn sa pa đã làm đơn khiếu nại nhiều nơi
numiennam-camhieu-001-001755-002716|báo tài nguyên và môi trường nhận được đơn kêu cứu khẩn cấp của ông phạm văn phượng * trú tại tổ bảy a , phố cầu mây , thị trấn sa pa , huyện sa pa lào cai
numiennam-camhieu-001-002730-003265|phản ánh về việc ủy ban nhân dân huyện sa pa cưỡng chế , thu hồi đất để mở đường trái quy định
numiennam-camhieu-001-003390-004323|chia sẻ với phóng viên báo tài nguyên và môi trường * ông phạm văn phượng cho biết * gia đình ông có bốn thửa đất đều thuộc tổ mười một bê thị trấn sa pa
numiennam-camhieu-001-004335-004692|do ông nhận chuyển nhượng mua lại của người dân trên địa bàn huyện
numiennam-camhieu-001-004788-005666|đến năm hai ngàn không trăm mười bốn . ủy ban nhân dân tỉnh lào cai có chủ trương thu hồi đất để nâng cấp cải tạo mở đường nguyễn chí thanh tại thị trấn sa pa
```
- Định dạng \<name\>|\<text\>, trong đó *name* là tên audio, *text* là nội dung của audio tương ứng
- Text là các chữ thuần việt viết thường
	- Các chữ số phải được viết dưới dạng chữ đúng theo cách đọc của PTV (VD: năm 2019 -> năm hai không mười chín)
	 - Các từ vay mượn hoặc viết tắt phải được viết đầy đủ theo cách đọc của PTV (VD: ronaldo -> rô nan đô, donald trump -> đô nan trăm)
- Có 3 loại dấu bao gồm , . * sẽ được đánh tùy theo cách ngắt nghỉ của PTV:
	- Đánh dấu **,** giữa 2 từ nếu khoảng silence giữa chúng khoảng **0.15 - 0.3s**
	- Đánh dấu **.** giữa 2 từ nếu khoảng silence giữa chúng khoảng **0.3 - 0.45s**
	- Đánh dấu **\*** giữa 2 từ nếu khoảng silence giữa chúng lớn hơn **0.45s**

Sử dụng Audacity để tự động tách các khoảng **speech** và **silence**:

- Chọn Audio cần cắt: File -> Import -> Audio
- Chọn toàn bộ audio vừa import: Ctrl + A
- Tự động tách vùng speech và silence: Analyze -> Sound Finder
- Chọn tham số cắt: 26.0 - 0.100 - 0.100 - 0.100 - 0
- Ấn OK để cắt
- Lưu thông tin cắt: File -> Export -> Export Labels  

Config file **hparams.py** bằng cách sửa các dòng config sau:
```
data='data/camhieu', # Đường dẫn đến folder data
version='v1-20191113', # Tên version training
...
sampling_rate=16000, # Sample rate của audio
```
	
### 1.3. Tiền xử lý dữ liệu

Sau khi đã chuẩn bị đầy đủ dữ liệu và nhãn như trên, ta sẽ tiến hành các bước tiền xử lý dữ liệu
#### Chuẩn hóa audio
Config file **hparams.py**
```
################################
# Audio Preprocess Parameters  #
################################
norm_volume=True, # True nếu muốn chuẩn hóa âm lượng audio
volume_ratio=0.7,
denoise=True, # True nếu muốn giảm noise cho audio, 
              # nếu audio không có noise thì ko nên bật
noise_frame=6,
vad=False, # True nếu muốn sử dụng Voice Acitivity Detection
vad_aggressiveness=1,
trim_silence=True, # True nếu muốn loại bỏ các khoảng silence lớn
trim_top_db=40,

filter_audios=True,  # True nếu muốn loại bỏ các audio dài
longest_wav_in_seconds=12, # Ngưỡng audio loại bỏ
limit_total_dur_in_hours=None,
```
Chạy:
```
python prepare_wavs.py --num_workers 16
```
Audio sau khi được chuẩn hóa sẽ được lưu vào folder **wavs_train**
```
ztacotron2
└───camhieu
	└───data
	    └───camhieu
	        │   metadata.txt
	        └───wavs_raw
	        │   │   numiennam-camhieu-001-000697-001659.wav
	        │   │   ...
	        └───wavs_train
	            │   numiennam-camhieu-001-000697-001659.wav
	            │   ...
```
#### Tách dữ liệu training & validation
Chạy:
```
python prepare_files_list.py
```
File metadata được tách làm 2 tập train và val, lưu ở thư mục **files_lists**
```
ztacotron2
└───camhieu
	└───data
	    └───camhieu
	        │   metadata.txt
	        │   wavs_raw
	        │   wavs_train
	        └───files_lists
	            │   metadata_train.txt
	            │   metadata_val.txt
```
### 1.4. Config feature
Sửa config file **[hparams.py](http://hparams.py/)**:
```
################################
# Audio Feature Parameters     #
################################
max_wav_value=32768.0,
sampling_rate=22050,
filter_length=1024,
hop_length=256,
win_length=1024,
n_mel_channels=80,
mel_fmin=55,
mel_fmax=7650.0,
```

| Gender | sampling_rate | hop_length | win_length | mel_fmin | mel_fmax |
|:------:|:-------------:|:----------:|:----------:|:--------:|:--------:|
|  Male  |     16000     |     200    |     800    |   55.0   |  7600.0  |
|  Male  |     22050     |     256    |    1024    |   55.0   |  7650.0  |
| Female |     16000     |     200    |     800    |   95.0   |  7600.0  |
| Female |     22050     |     256    |    1024    |   95.0   |  7600.0  |

### 1.5. Training
Chạy training trên GPU 0:
```
python train_tacotron_new.py --cuda 0
```
Sau khi chạy, thư mục lưu model và tensorboard log sẽ được lưu trong **experiments**:
```
ztacotron2
└───camhieu
	└───experiments
	    └───camhieu
	        └───logs
	        │   └───v1-20191113
	        │       │   events.out.tfevents.1573640376.ubuntu1804
	        └───tacotron_models
	            └───v1-20191113
	                │   tacotron2_0k.pt
```
Với config mặc định, cứ mỗi 5000 iter thì model sẽ được lưu lại 1 lần: tacotron2_0k.pt, tacotron2_5k.pt, tacotron2_10k.pt, ....

Nếu muốn tiếp tục training từ checkpoint cuối cùng, chạy:
```
python train_tacotron_new.py 
	-p=warm_start=false,checkpoint_path=last 
	--cuda 0
```
Nếu muốn tiếp tục training từ checkpoint bất kỳ, chạy:
```
python train_tacotron_new.py 
	-p=warm_start=false,checkpoint_path=/path/to/some_checkpoint.pt 
	--cuda 0
```

### 1.6. Kiểm tra log
Check log trên tensorboard:
```
tensorboard --logdir experiments/camhieu/logs
```

### 1.7. Tạo file TTS từ model
Gen audio với model mới nhất:
```
python gen_tts.py 
	--tacotron2 last 
	--waveglow waveglow_models/waveglow_doanngocle_v2 
	--test test_set/vlsp2018.txt 
	--cuda 0
```
Gen audio với model bất kỳ:
```
python gen_tts.py 
	--tacotron2 /path/to/some_checkpoint.pt 
	--waveglow waveglow_models/waveglow_doanngocle_v2 
	--test test_set/vlsp2018.txt 
	--cuda 0
```
Audio sẽ được gen ra tại thư mục **tts_samples**:
```
ztacotron2
└───camhieu
	└───experiments
	    └───camhieu
	        │   logs
	        │   tacotron_models
	        └───tts_samples
	            └───tacotron2_v1-20191113_5k+waveglow_doanngocle_v2
```

## 2. Waveglow

Cấu trúc project Waveglow
```
zwaveglow
└───base
	└───data
	│   │   camhieu
	│   │   ...
	└───experiments
	│   │   camhieu
	│   │   ...
	└───utils
	│   │   audio.py
	│   │   display.py
	│   │   file.py
	│   config.json
	│   distributed.py	
	│   gen_wavs.py 
	│   glow.py
	│   mel2samp.py
	│   paths.py
	│   plotting.py
	│   prepare_files_list.py
	│   stft.py
	│   train_waveglow.py
```

### 2.1. Môi trường training
Môi trường training đã được đóng gói và chuẩn bị sẵn dưới dạng conda enviroment
```
conda activate waveglow
```

### 2.2. Chuẩn bị dữ liệu training
Clone folder **base** để tạo 1 project riêng cho giọng mới
```
zwaveglow
└───base
└───camhieu
```

Thư mục chứa dữ liệu training cho từng giọng được đặt trong thư mục **data**, có cấu trúc như sau:
```
zwaveglow
└───camhieu
	└───data
	    └───camhieu
	        └───wavs
	            │   numiennam-camhieu-001-000697-001659.wav
	            │   numiennam-camhieu-001-001755-002716.wav
	            │   ...
```
**wavs** là thư mục chứa các đoạn audio đã được cắt nhỏ ra theo từng câu từ audio gốc, cần đảm bảo các tiêu chí sau:
- Có độ dài từ 2-10s
- Đầu và đuôi audio không được cắt lẹm vào từ
- Các audio có format WAV - PCM 16 bit, sample rate có thể là 16000Hz hoặc 22050Hz tùy vào mô hình huấn luyện
- Audio wavs có thể lấy từ dữ liệu training của tacotron 
- Tổng dữ liệu chỉ nên có thời lượng từ **4-5h**, không nên nhiều quá, và cũng không nên ít quá

Config file **config.json** bằng cách sửa các dòng config sau:
```
{
	"data": "data/camhieu, # Đường dẫn đến folder data
	"version": "v1-20191119", # Tên version training
	...
	"feature_config": {
		...
	    "sampling_rate": 22050, # Sample rate của audio
	    ...
	},
	...
}
```

### 2.3. Tiền xử lý dữ liệu
Chạy:
```
python prepare_files_list.py
```
Đường dẫn dữ liệu sẽ được tách làm 2 tập train và test, lưu thành các file như sau:
```
ztacotron2
└───camhieu
	└───data
	    └───camhieu
	        └───wavs
	        │   all.txt
	        │   train.txt
	        │   test.txt
```

### 2.4. Config feature
Sửa config file **config.json**:
```
"feature_config": {
    "segment_length": 16000,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 55.0,
    "mel_fmax": 7650.0
},
```

| Gender | sampling_rate | hop_length | win_length | mel_fmin | mel_fmax |
|:------:|:-------------:|:----------:|:----------:|:--------:|:--------:|
|  Male  |     16000     |     200    |     800    |   55.0   |  7600.0  |
|  Male  |     22050     |     256    |    1024    |   55.0   |  7650.0  |
| Female |     16000     |     200    |     800    |   95.0   |  7600.0  |
| Female |     22050     |     256    |    1024    |   95.0   |  7600.0  |

### 2.5. Training
Chạy training trên GPU 0:
```
python train_waveglow.py --cuda 0
```
Sau khi chạy, thư mục lưu model và tensorboard log sẽ được lưu trong **experiments**:
```
zwaveglow
└───camhieu
	└───experiments
	    └───camhieu
	        └───logs
	        │   └───v1-20191119
	        │       │   events.out.tfevents.1573640376.ubuntu1804
	        └───checkpoints
	            └───v1-20191113
	                │   waveglow_camhieu_v1-20191119_0k.pt
```
Với config mặc định, cứ mỗi 5000 iter thì model sẽ được lưu lại 1 lần: tacotron2_0k.pt, tacotron2_5k.pt, tacotron2_10k.pt, ....

Mặc định "checkpoint_path": "last" trong config, nên nếu muốn tiếp tục training từ checkpoint cuối cùng, chỉ cần chạy:
```
python train_waveglow.py --cuda 0
```
Nếu muốn tiếp tục training từ checkpoint bất kỳ, sửa **checkpoint_path** trong file **config.json** rồi chạy:
```
python train_waveglow.py --cuda 0
```
### 2.6. Kiểm tra log
Check log trên tensorboard:
```
tensorboard --logdir experiments/camhieu/logs
```
### 2.7. Tạo file TTS từ model
Gen audio với model mới nhất:
```
python gen_wavs.py --cuda 0
```
