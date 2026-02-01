from youtube_transcript_api import YouTubeTranscriptApi

def get_subtitles(url:str):
    """
    Returns the subtitles of a Youtube video as a string.
    
    Args:
        url: Url of the Youtube video
        example: "https://www.youtube.com/watch?v=CKS1glzmDVc"
    Returns:
        A dictionary with a "status" key indicating success or error, and a "subtitles" key containing the subtitles if successful.
        Success: {"status":"success","subtitles":<subtitles_string>}
        Error: {"status":"error","message":<error_message>}

    """
    # Extract the video id from url
    if url.startswith("https://youtu.be/"):
        vid_id = url.split("/")[-1]
    else:
        vid_id = url.split("v=")[-1].split("&")[0]


    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(vid_id)

    result = ""

    # is iterable
    for snippet in fetched_transcript:
        result+=snippet.text

    if not result:
        return {"status":"error","message":"No subtitles found"}
    return {"status":"success","subtitles":result}


if __name__ == "__main__":
    url = "https://youtu.be/eC8mZceIy5k"
    subtitles = get_subtitles(url)
    print(subtitles)
