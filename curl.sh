curl --request POST      --url https://us-west-2.recall.ai/api/v1/bot/      --header "Authorization: 869b9fa1b9db71e0aa30eb0d6e339356350d26c2"      --header "accept: application/json"      --header "content-type: application/json"      --data '
{
  "meeting_url": "https://meet.google.com/edn-crsu-grg",
  "recording_config": {
    "audio_mixed_raw": {},
    "realtime_endpoints": [
      {
        "type": "websocket",
        "url": "wss://dev.lili-o.com/ws?url=edn-crsu-grg",
        "events": ["audio_mixed_raw.data"]
      }
      
      ]
}}
'