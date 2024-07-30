import click
from flask.cli import with_appcontext
from db_models import app, db, MedicalPrediction

@with_appcontext
def create_database():
    db.create_all()
    print('Database is created.')

@click.command()
def create_db():
    with app.app_context():
        create_database()

app.cli.add_command(create_db)

if __name__ == '__main__':
    create_db()
