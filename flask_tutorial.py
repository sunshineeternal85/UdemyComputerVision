# %%
from flask import Flask, render_template, request



# %%
app = Flask(__name__)

# Your original greet1 function
def greet1(name: str):
    return 'hello ' + name + '!'

# Define a route for the home page ('/')
# This route will handle both GET requests (to display the form)
# and POST requests (to process the form submission)
@app.route('/', methods=['GET', 'POST'])
def index():
    greeting = None # Initialize greeting to None

    if request.method == 'POST':
        # If the request is a POST (form submitted)
        # Get the value from the 'user_input' textarea
        user_input = request.form['user_input']
        # Call your greet1 function
        greeting = greet1(user_input)

    # Render the HTML template, passing the 'greeting' variable to it
    return render_template('index.html', greeting=greeting)

# %%
# Run the Flask app
if __name__ == '__main__':
    # debug=True allows for automatic reloading on code changes and provides a debugger
    app.run(debug=True)
# %%
