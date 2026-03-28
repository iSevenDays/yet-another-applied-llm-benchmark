# Prompt snippet: buggy parse_transcript function shown to LLM.
# Used by: whisper_merge.py (question + header)
# This file is read as text, not imported as a module.
def parse_transcript(content):
    transcript_segments = []
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]  (.*)'
    matches = re.findall(pattern, content)

    for start_time, end_time, text in matches:
        start_seconds = timedelta(hours=int(start_time[:2]), minutes=int(start_time[3:5]), seconds=int(start_time[6:8]), milliseconds=int(start_time[9:])).total_seconds()
        end_seconds = timedelta(hours=int(end_time[:2]), minutes=int(end_time[3:5]), seconds=int(end_time[6:8]), milliseconds=int(end_time[9:])).total_seconds()
        transcript_segments.append((start_seconds, end_seconds, text))

    return transcript_segments
