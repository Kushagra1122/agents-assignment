import logging

import aiohttp
from dotenv import load_dotenv

from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram

logger = logging.getLogger("weather-example")
logger.setLevel(logging.INFO)

load_dotenv()


class WeatherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a weather agent.",
            llm=openai.realtime.RealtimeModel(
                turn_detection=None,  # Disable server-side turn detection
            ),
        )

    @function_tool
    async def get_weather(
        self,
        latitude: str,
        longitude: str,
    ):
        """Called when the user asks about the weather. This function will return the weather for
        the given location. When given a location, please estimate the latitude and longitude of the
        location and do not ask the user for them.

        Args:
            latitude: The latitude of the location
            longitude: The longitude of the location
        """

        logger.info(f"getting weather for {latitude}, {longitude}")
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        weather_data = {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # response from the function call is returned to the LLM
                    weather_data = {
                        "temperature": data["current"]["temperature_2m"],
                        "temperature_unit": "Celsius",
                    }
                else:
                    raise Exception(f"Failed to get weather data, status code: {response.status}")

        return weather_data


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room_name": ctx.room.name,
        "user_id": "your user_id",
    }

    # Configure session with smart interruption filtering
    session = AgentSession(
        # Use local VAD for turn detection
        vad=silero.VAD.load(),
        turn_detection="vad",
        # Add STT for transcript access during interruption filtering
        stt=deepgram.STT(
            model="nova-2",
            interim_results=True,
        ),
        # Configure smart interruption filtering
        ignore_words=[
            "yeah", "yes", "yep", "yup", "uh-huh", "uh huh", "uhuh",
            "okay", "ok", "k", "mm-hmm", "mmhmm", "mhm", "mm",
            "right", "sure", "got it", "gotcha", "i see",
            "hmm", "hm", "ah", "oh", "uh",
        ],
        interrupt_words=[
            "stop", "wait", "hold on", "pause", "cancel", "quit",
            "no", "nope", "don't", "stop talking", "be quiet",
            "shut up", "enough", "actually",
        ],
        interruption_grace_period=0.3,
    )

    # Add event handlers for conversation logging
    @session.on("user_input_transcribed")
    def on_user_input(ev):
        if ev.is_final:
            logger.info(f"[USER FINAL] {ev.transcript}")

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        logger.info(f"[STATE] Agent state: {ev.state}")

    await session.start(
        agent=WeatherAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
