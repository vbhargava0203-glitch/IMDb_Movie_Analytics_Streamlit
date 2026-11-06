"""Unit checks use invented fixture rows ONLY; never used as production movie statistics."""
import math
import unittest

import pandas as pd

from utils import (
    filter_movies, find_movie, format_usd, genre_counts,
    parse_currency, parse_multi_value, prepare_movies, top_gross, unique_values,
)


def make_fixture():
    return pd.DataFrame([
        {"primaryTitle":"Movie Alpha", "averageRating":9.1, "numVotes":1000000,
         "genres":"['Action', 'Drama']", "countriesOfOrigin":"['USA', 'UK']",
         "grossWorldwide":"$2.92B", "primaryImage":"https://example.org/alpha.png"},
        {"primaryTitle":"Movie Beta", "averageRating":8.0, "numVotes":12000,
         "genres":"['Drama']", "countriesOfOrigin":"['India']",
         "grossWorldwide":5000000.0, "primaryImage":"https://example.org/beta.png"},
        {"primaryTitle":"Movie Gamma", "averageRating":8.5, "numVotes":750000,
         "genres":"Action, Sci-Fi", "countriesOfOrigin":"USA, Canada",
         "grossWorldwide":"$1.2M", "primaryImage":""},
    ])


class MovieDataTests(unittest.TestCase):
    def setUp(self):
        self.raw = make_fixture()
        self.df = prepare_movies(self.raw)

    def test_multi_value_string_and_list(self):
        self.assertEqual(parse_multi_value("['Action', 'Drama']"), ["Action", "Drama"])
        self.assertEqual(parse_multi_value("USA, UK"), ["USA", "UK"])
        self.assertEqual(parse_multi_value(None), [])
        self.assertEqual(parse_multi_value([]), [])

    def test_currency(self):
        self.assertEqual(parse_currency("$2.92B"), 2920000000.0)
        self.assertEqual(parse_currency("1,000,000"), 1000000.0)
        self.assertTrue(math.isnan(parse_currency(None)))
        self.assertEqual(format_usd(2920000000), "$2.92B")

    def test_raw_unchanged_and_columns(self):
        self.assertEqual(self.raw.loc[0,"grossWorldwide"], "$2.92B")
        self.assertIn("genreList", self.df.columns)
        self.assertEqual(len(self.df), 3)

    def test_genre_explosion(self):
        counts = genre_counts(self.df)
        self.assertEqual(int(counts["Action"]), 2)
        self.assertEqual(int(counts["Drama"]), 2)
        self.assertEqual(int(counts["Sci-Fi"]), 1)

    def test_aggregations_and_sort(self):
        self.assertEqual(top_gross(self.df).iloc[0]["primaryTitle"], "Movie Alpha")
        self.assertEqual(int(self.df["numVotes"].sum()), 1762000)
        self.assertEqual(unique_values(self.df,"countryList"), ["Canada","India","UK","USA"])

    def test_all_filters_concurrently(self):
        result = filter_movies(
            self.df,
            genre="Action", countries=["USA"], rating_range=(8.0,9.0),
            min_votes=500000, title="Gamma", sort_order="Descending",
        )
        self.assertEqual(result["primaryTitle"].tolist(), ["Movie Gamma"])

    def test_sort_and_literal_title_search(self):
        result = filter_movies(self.df, rating_range=(0,10), sort_order="Ascending")
        self.assertEqual(result["primaryTitle"].tolist(), ["Movie Beta", "Movie Gamma", "Movie Alpha"])
        self.assertEqual(len(filter_movies(self.df,title="[",rating_range=(0,10))),0)

    def test_search_form_logic(self):
        movie, number = find_movie(self.df,title="Movie Alpha",genre="Action",minimum_rating=8.5)
        self.assertEqual(movie["primaryTitle"], "Movie Alpha")
        self.assertEqual(number, 1)
        movie, number = find_movie(self.df,title="Movie Alpha",genre="Sci-Fi",minimum_rating=7)
        self.assertIsNone(movie)
        self.assertEqual(number, 0)

    def test_missing_required_column(self):
        with self.assertRaisesRegex(ValueError,"grossWorldwide"):
            prepare_movies(self.raw.drop(columns=["grossWorldwide"]))


if __name__ == "__main__":
    unittest.main()
