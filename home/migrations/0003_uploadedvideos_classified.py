# Generated by Django 4.2.7 on 2023-11-28 13:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_uploadedvideos_violent'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedvideos',
            name='classified',
            field=models.BooleanField(default=False),
        ),
    ]
