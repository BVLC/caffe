#!/usr/bin/env python

"""
modelzoo is a CLI tool for listing

TASKS

[ ] integrate scripts/download_model_binary.py
[ ] integrate scripts/upload_model_to_gist.sh
"""

import os
import json
import click
import urllib2
import subprocess
import dateutil.parser


URL = "http://www.modelzoo.co/models"


@click.group()
def cli():
    "List and download publicly available Caffe models."
    pass


@cli.command()
def list():
    gists = json.load(urllib2.urlopen(URL + '.json'))
    for gist in gists:
        if gist is None:
            click.secho("Could not retrieve info", fg='red')
        else:
            click.secho("\n[{}]".format(gist['id']), fg='green', nl=False)
            click.echo("\t{}".format(gist['description']))
            click.echo("\towner: {}\tcreated: {}\tupdated: {}".format(
                gist['owner'],
                dateutil.parser.parse(gist['created_at']).strftime('%Y-%m-%d'),
                dateutil.parser.parse(gist['updated_at']).strftime('%Y-%m-%d')
            ))
            click.echo("\tgist: {}".format(gist['gist_id']))


@cli.command()
@click.argument('index_or_gist_id')
@click.option('--dirname', default='./models', help='where to download model')
def get(index_or_gist_id, dirname):
    model_dirname = os.path.join(dirname, 'modelzoo_' + index_or_gist_id)
    if os.path.exists(model_dirname):
        print("{} already exists! Make sure you're not overwriting you don't mean to.".format(model_dirname))
        exit(1)
    os.makedirs(model_dirname)

    print("Downloading model...")
    resource = urllib2.urlopen('{}/{}/download'.format(URL, index_or_gist_id))

    zip_filename = model_dirname + "/gist.zip"
    with open(zip_filename, 'wb') as file:
        file.write(resource.read())

    subprocess.call(['unzip', '-j', zip_filename, '-d', model_dirname])
    print("Extracted model to {}".format(model_dirname))

if __name__ == '__main__':
    cli()
