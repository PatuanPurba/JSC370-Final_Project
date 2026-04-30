import pandas as pd
import numpy as np
import json
from datetime import datetime
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import MultiLabelBinarizer

API_RATE = 40
MAX_WORKERS = 10

def getMovieObservation(input_dict):
    result = {}
    result["movie_id"] = input_dict["id"]
    result["imdb_id"] = input_dict["imdb_id"]
    result["title"] = input_dict["title"]
    result["original_title"] = input_dict["original_title"]
    result["language"] = input_dict["original_language"]
    result["popularity"] = input_dict["popularity"]
    result["vote_count"] = input_dict["vote_count"]
    result["rating"] = input_dict["vote_average"]
    result["release_date"] = input_dict["release_date"]
    result["runtime"] = input_dict["runtime"]
    result["budget"] = input_dict["budget"]
    result["revenue"] = input_dict["revenue"]
    result["cast_size"] = len(input_dict["credits"]["cast"])
    result["crew_size"] = len(input_dict["credits"]["crew"])

    genres = []
    for genre in input_dict["genres"]:
        genres.append(genre["name"])

    result["genres"] = genres
    return result


def getActObservation(input_dict, movie_id):
    result = {}
    result["id"] = input_dict["id"]
    result["movie_id"] = movie_id
    result["name"] = input_dict["name"]
    result["original_name"] = input_dict["original_name"]
    result["gender"] = input_dict["gender"]
    result["popularity"] = input_dict["popularity"]
    result["job"] = "Acting"

    return result


def getCrewObservation(input_dict, movie_id):
    result = {}
    result["id"] = input_dict["id"]
    result["movie_id"] = movie_id
    result["name"] = input_dict["name"]
    result["original_name"] = input_dict["original_name"]
    result["gender"] = input_dict["gender"]
    result["popularity"] = input_dict["popularity"]
    result["job"] = input_dict["job"]

    return result



def getMovieIDList(
        baseline_url,
        short_page_per_month, full_page_per_month,
        year_start, year_end
):
    with open("config/config.json", "r") as file:
        config = json.load(file)


    params = {
        "api_key": config["API_KEY"], 
        "sort_by": "vote_count.desc",
        "language": "en"
    }

    short_movie_ids = []
    full_movie_ids = []
    
    with requests.Session() as s:

        for year in range(year_start, year_end + 1):
            short_temp_list = []
            full_temp_list = []

            for month in range(1, 13):
                start_time = f"{year}-{month}-01"
                end_time = f"{year}-{month}-31"
                print(start_time)
            
                params["primary_release_date.gte"] = start_time
                params["primary_release_date.lte"] = end_time

                
                for i in range(1, short_page_per_month + 1):
                    params["page"] = i
                    response = s.get(baseline_url, params=params)

                    if response.status_code != 200:
                        print(json.dumps(response.json(), indent=4))
                        break
                    
                    response_result = response.json()["results"]
                    if not response_result:
                        break

                    else:
                        short_temp_list += [film["id"] for film in response_result]
                


                for i in range(1, full_page_per_month + 1):
                    params["page"] = i
                    response = s.get(baseline_url, params=params)

                    if response.status_code != 200:
                        print(json.dumps(response.json(), indent=4))
                        break
                    
                    response_result = response.json()["results"]
                    if not response_result:
                        break

                    else:
                        full_temp_list += [film["id"] for film in response_result]


            short_movie_ids += short_temp_list
            full_movie_ids += full_temp_list
        
    return short_movie_ids, full_movie_ids




def buildDataset(
        list_movie_id, type_movie, 
        cast_per_film,
        important_crew,
        add_person = False,
        movies_person_limit = 3000,
):
        with open("config/config.json", "r") as file:
            config = json.load(file)


        # Getting Movie Details and Cast Crew for each Movie ID
        error_movie_id = []
        params = {
            "api_key" : config["API_KEY"],
            "language": "en_US",
            "append_to_response" : "credits"
        }
 

        results_movie = []
        results_person = []
        index = 0

        with requests.Session() as s:
            
            for i, movie_id in enumerate(list_movie_id):
                response = s.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params=params, timeout=10)

                if response.status_code != 200:
                    error_movie_id.append(movie_id)
                    time.sleep(1/API_RATE)
                
                else:
                    results_movie.append(getMovieObservation(response.json()))

                    if add_person:
                        credits = response.json()["credits"]
                        cast = credits["cast"]
                        results_person += [getActObservation(cast[j], movie_id) for j in range(min(len(cast), cast_per_film))]

                        crew = credits["crew"]
                        results_person += [getCrewObservation(c, movie_id) for c in crew if c.get("job") in important_crew]
                    
            
                if add_person and (i + 1) % movies_person_limit == 0: 
                    print(len(results_person))
                    pd.DataFrame(results_person).to_csv(f"data/raw/person_{index}.csv")
                    index += 1
                    results_person = []

        if add_person:
            pd.DataFrame(results_person).to_csv(f"data_1/person_{index}.csv")
        
        movies = pd.DataFrame(results_movie)

        mlb = MultiLabelBinarizer()
        genre_encoded = pd.DataFrame(
            mlb.fit_transform(movies["genres"]),
            columns=mlb.classes_,
            index=movies.index
        ).add_prefix("genre_")

        df_encoded = pd.concat([movies, genre_encoded], axis=1)
        df_encoded.to_csv(f"data_1/{type_movie}.csv")

        
        # Try for Multithreading, however seems doesn't work since the heavy part is on 
        #
        # def request_movie_detail(movie_id):
        #     return requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params=params, timeout=10), movie_id
        # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = [executor.submit(request_movie_detail(movie_id), movie_id) for movie_id in list_movie_id]

        #     for future in as_completed(futures):
        #         try:
        #             response, movie_id = future.result()
                    
        #             if response.status_code == 200:
        #                 results.append(getMovieObservation(response.json()))

        #             elif response.status_code == 429:
        #                 error_movie_id.append(movie_id)
        #                 time.sleep(1/API_RATE)
                
        #         except Exception:
        #             continue
