from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import getpass
import os
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel, Field

import subprocess
import json
import pickle

from typing import List

class Segment(BaseModel):
  start_time: float = Field(..., description="The start time of the segment in seconds")
  end_time: float = Field(..., description="The end time of the segment in seconds")
  yt_title: str = Field(..., description="The youtube title to make this segment as a viral sub-topic")
  description: str = Field(..., description="The detailed youtube description to make this segment viral ")
  duration: int = Field(..., description="The duration of the segment in seconds")

class VideoTranscript(BaseModel):
  segments: List[Segment] = Field(..., description="List of viral segments in the video")

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

youtube_url = "https://www.youtube.com/watch?v=AIqJm3bxybg&pp=ygUGcmVkZGl0"
yt = YouTube(youtube_url)
video = yt.streams.filter(file_extension='mp4').first()
safe_title = ""
absolute_path = os.environ.get("ABSOLUTE_PATH")

llm = ChatOpenAI(model='gpt-4o-mini',
                 temperature=0.7,
                 max_tokens=None,
                 timeout=None,
                 max_retries=2
                 )
def download_video(title):
    global safe_title
    safe_title = "".join(ch for ch in title if ch.isalnum() or ch == " ").replace(" ", "_")
    video.download(output_path=f"{absolute_path}/downloaded_videos", filename=f"{safe_title}.mp4")
    write_transcript()

def write_transcript():
    video_id = yt.video_id
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    with open(f"generated_clips/transcript.txt", "w+") as file:
        file.write(str(transcript))

def get_transcript():
    with open("generated_clips/transcript.txt", "r") as file:
        return file.read()


def get_segments():
    prompt = f"""Provided to you is a transcript of a video. 
    Please identify all segments that can be extracted as 
    subtopics from the video based on the transcript.
    Make sure each segment is strictly between 60-500 seconds in duration.
    Make sure you provide extremely accruate timestamps
    and respond only in the format provided. 
    \n Here is the transcription : \n {get_transcript()}"""


    messages = [
        {"role": "system", "content": "You are a viral content producer. You are master at reading youtube transcripts and identifying the most intriguing content. You have extraordinary skills to extract subtopic from content. Your subtopics can be repurposed as a separate video."},
        {"role": "user", "content": prompt}
    ]

    structured_llm = llm.with_structured_output(VideoTranscript)
    ai_msg = structured_llm.invoke(messages)
    print(ai_msg)
    parsed_content = ai_msg.model_dump()['segments']
    save_parsed_content(parsed_content)

def save_parsed_content(parsed_content: dict):
    with open("parsed_content.pkl", "wb+") as bfile:
        pickle.dump(parsed_content, bfile)

def slice_video(parsed_content):
    segment_labels = []
    video_title = safe_title


    for i, segment in enumerate(parsed_content):
        start_time = segment['start_time']
        end_time = segment['end_time']
        yt_title = segment['yt_title']
        description = segment['description']
        duration = segment['duration']

        output_file = f"generated_clips/{video_title}_{str(i+1)}.mp4"
        command = f"ffmpeg -i downloaded_videos/{safe_title}.mp4 -ss {start_time} -to {end_time} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_file}"
        subprocess.call(command, shell=True)
        segment_labels.append(f"Sub-Topic {i+1}: {yt_title}, Duration: {duration}s\nDescription: {description}\n")

    with open('generated_clips/segment_labels.txt', 'w') as f:
        for label in segment_labels:
            f.write(label +"\n")

    # save the segments to a json file
    with open('generated_clips/segments.json', 'w') as f:
        json.dump(parsed_content, f, indent=4)

if __name__ == "__main__":
    download_video("Video_name")
    get_segments()
    with open("parsed_content.pkl", "rb") as bfile:
        content = pickle.load(bfile)
        slice_video(content)
