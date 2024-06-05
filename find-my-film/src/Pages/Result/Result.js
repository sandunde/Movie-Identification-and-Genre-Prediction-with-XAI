import React, { useState, useEffect } from "react";
import axios from "axios";
import Button from "react-bootstrap/Button";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Spinner from "react-bootstrap/Spinner";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSearch } from "@fortawesome/free-solid-svg-icons";
import "./Result.css";
import { useNavigate } from "react-router";

const Result = () => {
  const [movieData, setMovieData] = useState(null);
  const [predictedGenre, setPredictedGenre] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [noSubtitle, setNoSubtitle] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [genresPerPage] = useState(10);
  const apiKey = "1a7669b3";
  const movieTitle = localStorage.getItem("movieName");
  const navigate = useNavigate();

  const handleGenreClick = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/predict-genre", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ movieTitle }),
      });

      if (!response.ok) {
        throw new Error("Failed to get predicted genres");
      }

      const data = await response.json();
      setPredictedGenre(data.movie_genre);
      console.log(predictedGenre);
      if (predictedGenre === null) {
        setNoSubtitle(true);
      }
    } catch (error) {
      console.error("Error fetching predicted genres:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate("/home");
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          `http://www.omdbapi.com/?t=${movieTitle}&apikey=${apiKey}`
        );
        setMovieData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, [apiKey, movieTitle]);

  const handleSearch = () => {
    localStorage.setItem("movieName", searchQuery);
    setMovieData(null);
  };

  const indexOfLastGenre = currentPage * genresPerPage;
  const indexOfFirstGenre = indexOfLastGenre - genresPerPage;
  const currentGenres = predictedGenre.slice(
    indexOfFirstGenre,
    indexOfLastGenre
  );

  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <Container fluid>
      <div className="nav-bar">
        <Row>
          <Col>
            <h3 onClick={handleBack}>FindMyFilm</h3>
          </Col>
          <Col>
            <div className="search-container">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter movie title"
                className="search-input"
              />
              <FontAwesomeIcon
                icon={faSearch}
                className="search-icon"
                onClick={handleSearch}
              />
            </div>
          </Col>
          <Col className="d-flex justify-content-end mt-1 login-btn">
            <Button className="mr-2" variant="outline-dark">
              Watch on Netflix
            </Button>
            <Button variant="outline-dark">Apple TV</Button>
          </Col>
        </Row>
      </div>
      {movieData ? (
        <div>
          <Row>
            <Col>
              <h2 className="movie-title">{movieData.Title}</h2>
            </Col>
          </Row>
          <Row>
            <Col sm={6}>
              <img src={movieData.Poster} alt={movieData.Title} />
            </Col>
            <Col sm={6}>
              <p>
                <strong>Year:</strong> {movieData.Year}
              </p>
              <p>
                <strong>Language:</strong> {movieData.Language}
              </p>
              <p>
                <strong>Genre:</strong> {movieData.Genre}
              </p>
              <p>
                <strong>Director:</strong> {movieData.Director}
              </p>
              <p>
                <strong>Actors:</strong> {movieData.Actors}
              </p>
              <p>
                <strong>Plot:</strong> {movieData.Plot}
              </p>
              <p>
                <strong>MPAA Rating:</strong> {movieData.Rated}
              </p>
              <Button onClick={handleGenreClick} variant="outline-dark">
                Genre Prediction
              </Button>
              {loading && (
                <Spinner animation="border" role="status" className="spinner" />
              )}
            </Col>
          </Row>
          <div className="genre-column">
            {currentGenres.length > 0 ? (
              <div>
                <h3>Predicted Genres:</h3>
                <ul>
                  {currentGenres.map((genre, index) => (
                    <li key={index}>
                      <strong>Scene {indexOfFirstGenre + index + 1}:</strong>{" "}
                      {genre[0]} - Confidence: {genre[1].toFixed(4)}
                    </li>
                  ))}
                </ul>
                <ul className="pagination">
                  <ul className="pagination">
                    {Array.from({
                      length: Math.min(
                        5,
                        Math.ceil(predictedGenre.length / genresPerPage)
                      ),
                    }).map((_, index) => (
                      <li key={index} className="page-item">
                        <Button
                          onClick={() => paginate(index + 1)}
                          className="page-link"
                        >
                          {index + 1}
                        </Button>
                      </li>
                    ))}
                  </ul>
                </ul>
              </div>
            ) : (
              <div></div>
            )}
          </div>
          {noSubtitle && <div>Could not find</div>}
        </div>
      ) : (
        <h2 className="movie-title">{movieTitle}</h2>
      )}
    </Container>
  );
};

export default Result;
