from flask import Flask, render_template, request
from rec_sys import RecSys  # Ensure this is correctly imported from your recommendation system script

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        ingredients = request.form.get('ingredients')
        spice_level = request.form.get('spice_level')
        cuisine_type = request.form.get('cuisine_type') or None
        if ingredients and spice_level:
            try:
                recommendations = RecSys(ingredients.split(', '), spice_level, cuisine_type, N=5)
            except Exception as e:
                recommendations = []
                print(e)  # For debugging
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
