import sys; sys.path.append('.')
from flask import Flask, redirect, render_template, request, url_for
from src.leg import simulate

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def login():
    move_name = ['b_move', 'g_move', 'r_move', 'y_move', 'w_move']
    move_value = []
    loc_name = ['b_loc', 'g_loc', 'r_loc', 'y_loc', 'w_loc']
    loc_value = []

    if request.method == 'POST':
        print(request.form)
        for i in move_name: # Upload move infomation
            if i in request.form:
                move_value.append(True)
            else:
                move_value.append(False)

        for i in loc_name: # Upload location information
            loc = request.form[i]
            loc_value.append(loc[5:])

        # Simulation
        simulate(move_value, loc_value)

        return redirect(url_for('result'))
    else:
        return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    