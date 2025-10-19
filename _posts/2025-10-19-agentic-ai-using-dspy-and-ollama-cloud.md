---
layout: post
title: Agentic AI using DSPy and Ollama Cloud
date: 2025-10-19 04:22 -0500
---
Putting it all together: A customer service airline booking Agent using DSPy + Ollama cloud models
The code below is based on https://dspy.ai/tutorials/customer_service_agent/ and modified to use Ollama cloud model.

Setup
0. pip install -qU dspy pydantic
1. Install Ollama from https://ollama.com/ 
2. Sign up for an Ollama Cloud account and get your API key.
3. Install the gpt-oss:120b-cloud model: Once we signin to ollama and have associated the laptop device with ollama user ID then we can connect to cloud model hosted at ollama.com by connecting to locally running ollama service. This enables running faster inference in cloud instead of running it locally on a model downloaded to laptop.
With dspy we can easily switch between cloud hosted and locally hosted models by setting dspy.configure(lm=<your-choice-of-model>)

ollama signin
ollama run gpt-oss:120b-cloud

4. Test if the setup is working:
import dspy

# Configure cloud Ollama (replace with your cloud endpoint)
cloud_ollama = dspy.LM(
    api_base="http://localhost:11434",
    api_key="",  
    model="ollama_chat/gpt-oss:120b-cloud",
    model_type="chat"
)

dspy.configure(lm=cloud_ollama)
print(cloud_ollama("What is the capital of France?"))
Simple airline customer service agent
	•	Book new trips on behalf of the user.
	•	Modify existing trips, including flight change and cancellation.
	•	On tasks it cannot handle, raise a customer support ticket.
Create Test Data and define the tools
Setup test data and tool calls for the agent.

from pydantic import BaseModel

class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight

class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile

user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",  # DSPy Airline 123
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}

Define Tools
	•	Have a docstring which defines what the tool does. If the function name is self-explanable, then you can leave the docstring empty.
	•	Have type hint for the arguments, which is necessary for LM to generate the arguments in the right format.

We need to prepare a list of tools so that the agent can behave like a human airline service agent:
	•	fetch_flight_info: get flight information for certain dates.
	•	pick_flight: pick the best flight based on some criteria.
	•	book_flight: book a flight on behalf of the user.
	•	fetch_itinerary: get the information of a booked itinerary.
	•	cancel_itinerary: cancel a booked itinerary.
	•	get_user_info: get users' information.
	•	file_ticket: file a backlog ticket to have human assist.


import random
import string


def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    if len(flights) == 0:
        raise ValueError("No matching flight found!")
    return flights


def fetch_itinerary(confirmation_number: str):
    """Fetch a booked itinerary information from database"""
    return itinery_database.get(confirmation_number)


def pick_flight(flights: list[Flight]):
    """Pick up the best flight that matches users' request. we pick the shortest, and cheaper one on ties."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]


def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def book_flight(flight: Flight, user_profile: UserProfile):
    """Book a flight on behalf of the user."""
    confirmation_number = _generate_id()
    while confirmation_number in itinery_database:
        confirmation_number = _generate_id()
    itinery_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinery_database[confirmation_number]


def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Cancel an itinerary on behalf of the user."""
    if confirmation_number in itinery_database:
        del itinery_database[confirmation_number]
        return
    raise ValueError("Cannot find the itinerary, please check your confirmation number.")


def get_user_info(name: str):
    """Fetch the user profile from database with given name."""
    return user_database.get(name)


def file_ticket(user_request: str, user_profile: UserProfile):
    """File a customer support ticket if this is something the agent cannot handle."""
    ticket_id = _generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id

Create ReAct Agent
We need to provide a signature to dspy.ReAct to define task, and the inputs and outputs of the agent, and tell it about the tools it can access.

import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fulfill users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )

agent = dspy.ReAct(
    DSPyAirlineCustomerService,
    tools = [
        fetch_flight_info,
        fetch_itinerary,
        pick_flight,
        book_flight,
        cancel_itinerary,
        get_user_info,
        file_ticket,
    ]
)

Use the Agent
	•	Invoke the dspy.ReAct with the inputs field defined in the signature, and it will start the reasoning and acting loop behind the scene.

# Configure DSPy to use cloud Ollama model
cloud_ollama = dspy.LM(
    api_base="http://localhost:11434",
    api_key="",  
    model="ollama_chat/gpt-oss:120b-cloud",
    model_type="chat"
)

dspy.configure(lm=cloud_ollama)
result = agent(user_request="please help me book a flight from SFO to JFK on 9/01/2025, my name is Adam")
print(result) #— Agent Trajectory shown below

print(itinery_database)
dspy.inspect_history(n=10)
confirmation_number = "{copy the confirmation number here}"

result = agent(user_request=f"i want to take DA125 instead on 09/01, please help me modify my itinerary {confirmation_number}")
print(result)

Agent Trajectory
    trajectory={'thought_0': "I need to retrieve Adam's user profile before I can book a flight for him. I'll start by fetching his user information from the database.", 'tool_name_0': 'get_user_info', 'tool_args_0': {'name': 'Adam'}, 'observation_0': UserProfile(user_id='1', name='Adam', email='adam@gmail.com'), 'thought_1': "Now that I have Adam's user profile, I need to find available flights from SFO to JFK on September 1, 2025. I will fetch flight information for that route and date.", 'tool_name_1': 'fetch_flight_info', 'tool_args_1': {'origin': 'SFO', 'destination': 'JFK', 'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}}, 'observation_1': [Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0), Flight(flight_id='DA125', date_time=Date(year=2025, month=9, day=1, hour=7), origin='SFO', destination='JFK', duration=9.0, price=500.0)], 'thought_2': 'I have the list of available flights. I will use the pick_flight tool to select the best option based on shortest duration and lowest price.', 'tool_name_2': 'pick_flight', 'tool_args_2': {'flights': [{'flight_id': 'DA123', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 1}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 3.0, 'price': 200.0}, {'flight_id': 'DA125', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 7}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 9.0, 'price': 500.0}]}, 'observation_2': Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0), 'thought_3': 'I have selected the best flight (DA123) for Adam. The next step is to book this flight using his user profile.', 'tool_name_3': 'book_flight', 'tool_args_3': {'flight': {'flight_id': 'DA123', 'date_time': {'year': 2025, 'month': 9, 'day': 1, 'hour': 1}, 'origin': 'SFO', 'destination': 'JFK', 'duration': 3.0, 'price': 200.0}, 'user_profile': {'user_id': '1', 'name': 'Adam', 'email': 'adam@gmail.com'}}, 'observation_3': ('ta18vp18', Itinerary(confirmation_number='ta18vp18', user_profile=UserProfile(user_id='1', name='Adam', email='adam@gmail.com'), flight=Flight(flight_id='DA123', date_time=Date(year=2025, month=9, day=1, hour=1), origin='SFO', destination='JFK', duration=3.0, price=200.0))), 'thought_4': 'The flight has been successfully booked for Adam. I will now finish the task.', 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},

DSPy ReAct module makes it simple to write an agent by just pointing to the tools.
