# Generated by Django 4.2.7 on 2024-01-17 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0009_alter_uploadedvideos_duration_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedvideos',
            name='duration',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=6),
        ),
        migrations.AlterField(
            model_name='uploadedvideos',
            name='progress',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=6),
        ),
    ]
