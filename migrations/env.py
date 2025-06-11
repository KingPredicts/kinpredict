import logging
from logging.config import fileConfig

from flask import current_app

from alembic import context
from sqlalchemy import pool # Make sure pool is imported if used, or engine_from_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None: # Ensure config_file_name is set
    fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

# <<< START OF CHANGES >>>
# Import your Flask app's db object and set target_metadata
# Assuming your main Flask app file is app5.py and your SQLAlchemy object is db
from app5 import db
target_metadata = db.metadata
# <<< END OF CHANGES >>>

# The following lines that try to get engine URL and set it in config
# are usually handled by Flask-Migrate when it sets up the context.
# If you're using Flask-SQLAlchemy, Flask-Migrate's default `env.py`
# should correctly get the database URL from your Flask app's config.
# So, these might be redundant or specific to older/different setups.
# Let's keep them for now but be aware Flask-Migrate typically handles this.

def get_engine():
    try:
        # this works with Flask-SQLAlchemy<3 and Alchemical
        return current_app.extensions['migrate'].db.get_engine()
    except (TypeError, AttributeError):
        # this works with Flask-SQLAlchemy>=3
        return current_app.extensions['migrate'].db.engine

def get_engine_url():
    try:
        return get_engine().url.render_as_string(hide_password=False).replace(
            '%', '%%')
    except AttributeError:
        return str(get_engine().url).replace('%', '%%')

# Set the sqlalchemy.url in the config. This is crucial for Alembic to know
# which database to connect to, especially in offline mode or if not using
# current_app context directly in all migration paths.
# Flask-Migrate usually helps configure this.
db_url = current_app.config.get('SQLALCHEMY_DATABASE_URI')
if db_url:
    config.set_main_option('sqlalchemy.url', db_url.replace('%', '%%'))
else:
    # Fallback if not in app config, try to get from engine
    # This part might be optional if your Flask app config always has SQLALCHEMY_DATABASE_URI
    logger.warning("SQLALCHEMY_DATABASE_URI not found in Flask app config, trying get_engine_url()")
    config.set_main_option('sqlalchemy.url', get_engine_url())


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# This get_metadata function might be overly complex if target_metadata is already set correctly above.
# Using the directly imported `target_metadata` is usually sufficient.
# def get_metadata():
#     if hasattr(current_app.extensions['migrate'].db, 'metadatas'):
#         return current_app.extensions['migrate'].db.metadatas[None]
#     return current_app.extensions['migrate'].db.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata, # Use the globally set target_metadata
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True # Add for SQLite offline consistency
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # this callback is used to prevent an auto-migration from being generated
    # when there are no changes to the schema
    # reference: http://alembic.zzzcomputing.com/en/latest/cookbook.html
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    # Flask-Migrate might set configure_args. Ensure we don't overwrite essential ones.
    conf_args = current_app.extensions['migrate'].configure_args
    if conf_args.get("process_revision_directives") is None:
        conf_args["process_revision_directives"] = process_revision_directives

    # <<< START OF CHANGES for render_as_batch >>>
    # Ensure render_as_batch is included, especially for SQLite
    # It's safer to add it directly if not already present in conf_args
    if 'render_as_batch' not in conf_args:
        # Check if using SQLite
        engine_url_str = str(get_engine().url)
        if engine_url_str.startswith('sqlite'):
            conf_args['render_as_batch'] = True
            logger.info("SQLite detected, 'render_as_batch=True' enabled for online migrations.")
    # <<< END OF CHANGES for render_as_batch >>>

    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata, # Use the globally set target_metadata
            **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    logger.info("Running migrations in offline mode...")
    run_migrations_offline()
else:
    logger.info("Running migrations in online mode...")
    run_migrations_online()