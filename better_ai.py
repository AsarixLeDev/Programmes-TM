from datetime import datetime
from pytz import timezone


def get_greeting_based_on_time(user_timezone='UTC'):
	user_tz = timezone(user_timezone)
	current_hour = datetime.now(user_tz).hour

	if current_hour < 12:
		return "Good morning!"
	elif 12 <= current_hour < 18:
		return "Good afternoon!"
	else:
		return "Good evening!"


def simple_ai_chatbot():
	responses = {
		"greetings": ["hello", "hi", "hey", "greetings", "salutations"],
		"wellbeing": ["how are you", "how are you doing", "how's it going"],
		"identity": ["what is your name", "who are you"],
		"farewell": ["exit", "quit", "goodbye"],
		"time": ["what time is it", "tell me the time", "current time"],
		"math": ["solve", "calculate", "math"]
	}

	while True:
		try:
			user_input = input("You: ").lower()
			if any(greeting in user_input for greeting in responses["greetings"]):
				print(f"AI: {get_greeting_based_on_time()} How can I assist you today?")
			elif any(wellbeing in user_input for wellbeing in responses["wellbeing"]):
				print("AI: I'm just a program, so I don't have feelings, but I'm here to assist you!")
			elif any(identity in user_input for identity in responses["identity"]):
				print("AI: I am an enhanced AI chatbot created to assist with various tasks.")
			elif any(farewell in user_input for farewell in responses["farewell"]):
				print("AI: Goodbye!")
				break
			elif any(time in user_input for time in responses["time"]):
				print(f"AI: The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
			elif any(math in user_input for math in responses["math"]):
				expression = input("AI: Please enter the math expression to solve: ")
				try:
					result = eval(expression)
					print(f"AI: The result is {result}.")
				except Exception as e:
					print(f"AI: An error occurred while calculating: {e}")
			else:
				print("AI: I'm sorry, I don't understand that command. Can you try something else?")
		except Exception as e:
			print(f"AI: An error occurred: {e}")


if __name__ == "__main__":
	# To run the chatbot
	simple_ai_chatbot()