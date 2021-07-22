# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2

import shutil


def convert_everything_to_mp4():
    cmd = 'bash scripts/swf2mp4.sh'

    os.system(cmd)


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    """
    frame_array: list, 영상으로 만들 대상 프래임 리스트
    path_out: str, 영상이 저장될 경로 및 영상 제목
    size: (int, int), 영상 해상도
    fps=25: int, frame rate, 기본값 25
    """

    # MPEG-4 포맷의 VideoWriter 객체 생성
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # 프래임 리스트를 영상으로 저장
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    """
    src_video_path: str, 원본 영상 경로
    start_frame: int, 시작 프래임
    end_frame: int, 끝 프래임
    """

    # 위에서 선언한 함수를 사용해 영상에서 모든 프래임 추출한 리스트 저장
    frames = video_to_frames(src_video_path)

    # 추출한 프래임에서 인수로 받은 영역만 반환
    return frames[start_frame: end_frame+1]


def extract_all_yt_instances(content):
    """
    content: json, 모든 데이터 정보가 들어있는 json 파일
    """

    # 변환한 영상 count
    cnt = 1

    if not os.path.exists('videos'):
        os.mkdir('videos')

    for entry in content:  # 각 gloss별 반복
        instances = entry['instances']  # gloss별 영상 정보

        for inst in instances:  # gloss 내부의 영상 정보 하나씩 반복
            url = inst['url']
            video_id = inst['video_id']

            # 영상의 출처가 youtube일 경우
            if 'youtube' in url or 'youtu.be' in url:
                cnt += 1

                yt_identifier = url[-11:]  # 유튜브 영상 코드

                # 원본과 타겟 파일명
                src_video_path = os.path.join(
                    'raw_videos_mp4', yt_identifier + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if not os.path.exists(src_video_path):
                    continue

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                # jsin 파일의 인덱스가 1부터이기 때문에 1씩 차감
                start_frame = inst['frame_start'] - 1
                end_frame = inst['frame_end'] - 1

                # 영상의 끝까지 사용할 경우 바로 카피
                if end_frame <= 0:
                    shutil.copyfile(src_video_path, dst_video_path)
                    continue

                # 원본 영상에서 원하는 범위만큼 프래임으로 추출
                selected_frames = extract_frame_as_video(
                    src_video_path, start_frame, end_frame)

                # when OpenCV reads an image, it returns size in (h, w, c)
                # when OpenCV creates a writer, it requres size in (w, h).
                # shape -> (h, w, c)
                # shape[:2][::-1] -> (w, h)
                size = selected_frames[0].shape[:2][::-1]

                # 해당 프래임들을 영상으로 저장
                convert_frames_to_video(selected_frames, dst_video_path, size)

                # 변환 count와 변환한 video 파일 출력
                print(cnt, dst_video_path)

            # 영상의 출처가 youtube가 아닐 경우 단순 복사
            else:
                cnt += 1

                src_video_path = os.path.join(
                    'raw_videos_mp4', video_id + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                if not os.path.exists(src_video_path):
                    continue

                print(cnt, dst_video_path)
                shutil.copyfile(src_video_path, dst_video_path)


def main():
    # 1. Convert .swf, .mkv file to mp4.
    convert_everything_to_mp4()

    # 모든 데이터의 정보를 담은 json 파일
    content = json.load(open('WLASL_v0.3.json'))

    # raw data 전처리 후 copy
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()
