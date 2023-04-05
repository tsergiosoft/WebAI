from flask import Flask, render_template, request, redirect, url_for
import os
import brain_app

app = Flask(__name__)

@app.route('/', methods=('GET', 'POST'))
@app.route('/index', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        if 'file1' in request.files:
            file = request.files['file1']
        if 'file2' in request.files:
            file = request.files['file2']
        if 'file3' in request.files:
            file = request.files['file3']
        if file:
            xfile = os.path.join('static', 'upl_' + file.name + '.jpg')
            print(xfile)
            current_file_directory = os.path.dirname(__file__)
            file_path = os.path.join(current_file_directory, "static/"+'upl_' + file.name + '.jpg')
            xfile = os.path.normpath(file_path)
            file.save(xfile)
    return render_template('index.html', res1=brain.res1, res2=brain.res2, res3=brain.res3, learning=brain.learn_active)

@app.route('/analyze_images/', methods=['POST'])
def analyze():
    print('analyze_images()')
    brain.analyze_uploaded_images()
    return redirect(url_for('index'))

@app.route('/learn_model/', methods=['POST'])
def learn():
    brain.start_learning()
    return render_template('index.html', res1=brain.res1, res2=brain.res2, res3=brain.res3, learning=brain.learn_active)

@app.route('/about')
def about():
    return render_template('about.html')

brain = brain_app.brain_AI() #create AI brain class for using by Application

# Run built - in development server that is designed to be used
# for testing and development purposes.
# Does not need when deploying a Flask application to a real web server
app.run() #Run built - in development server

# if __name__ == '__main__':
#     brain = brain_app.brain_AI() #create AI brain class for using by Application
#     app = MyApp() #create Application
#     app.run() #Run built - in development server

