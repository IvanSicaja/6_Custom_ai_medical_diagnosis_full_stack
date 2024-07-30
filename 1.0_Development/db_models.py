from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class MedicalPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    predicted_class = db.Column(db.Integer, nullable=True)
    predicted_probability = db.Column(db.String(10), nullable=False)

    def __repr__(self):
        return f'<MedicalPrediction {self.id}>'
