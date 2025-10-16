# import packages
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
from io import BytesIO
from PIL import Image

def reload_model():
    return None

def recommend_companies(df, user_query, top=7):
    """
    :param df: uploaded job
    :param user_query: string
    :return: list of dicts with keys: company, job_title, similarity, explanation
    """

    res = []
    for i in range(min(top, 7)):
        res.append({
            "Company":
        })