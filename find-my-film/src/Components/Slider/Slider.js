import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Button from "react-bootstrap/Button";
import Spinner from "react-bootstrap/Spinner";
import Mainlogo from "../../Assets/mainLogo.png";
import Form from "react-bootstrap/Form";
import "./Slider.css";

const Slider = () => {
  const [waveActive, setWaveActive] = useState(false);
  const [recognizedText, setRecognizedText] = useState("");
  const [movieName, setMovieName] = useState("");
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isTextInputMode, setIsTextInputMode] = useState(false);
  const [textInput, setTextInput] = useState("");
  const navigate = useNavigate();

  const handleButtonClick = async () => {
    setMovieName("");
    setRecognizedText("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      setMediaRecorder(recorder);
      const chunks = [];
      recorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: "audio/webm" });
        await sendAudioFile(audioBlob);
        setWaveActive(false);
      };

      recorder.start();

      setTimeout(() => {
        recorder.stop();
        setWaveActive(false);
        setLoading(true);
      }, 7000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
    setWaveActive(true);
  };

  const handleCancelClick = () => {
    window.location.reload();
  };

  const sendAudioFile = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recorded_audio.webm");

    try {
      const response = await fetch("http://localhost:5000/process_audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload audio file");
      }

      const data = await response.json();
      setRecognizedText(data.recognized_text);
      setMovieName(data.movie_name);
    } catch (error) {
      console.error("Error uploading audio file:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleTextInputSubmit = async () => {
    setLoading(true)
    setMovieName("");
    setRecognizedText("");
    try {
      const response = await fetch("http://localhost:5000/text-submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json" 
        },
        body: JSON.stringify({ textInput }),
      });

      if (!response.ok) {
        throw new Error("Failed to upload the text");
      }

      const data = await response.json();
      setRecognizedText(data.recognized_text);
      setMovieName(data.movie_name);
    } catch (error) {
      console.error("Error uploading the text:", error);
    } finally {
      setLoading(false);
    }
  };


  useEffect(() => {
    if (movieName !== "Couldn’t quite catch that" && movieName !== '') {
      localStorage.setItem("movieName", movieName);
      localStorage.setItem("recognizedText", recognizedText);
      navigate("/result");
    }
  }, [movieName, recognizedText, navigate]);

  return (
    <div className="slider-container">
      {!isTextInputMode && (
        <Button
          variant="outline-dark"
          className={`main-button ${waveActive ? "wave" : ""}`}
          onClick={handleButtonClick}
          disabled={waveActive}
        >
          Audio
        </Button>
      )}
      {waveActive && (
        <Button
          variant="danger"
          className="cancel-button"
          onClick={handleCancelClick}
        >
          CANCEL
        </Button>
      )}
      {loading && (
        <Spinner animation="border" role="status" className="spinner" />
      )}
      {!isTextInputMode && movieName && (
        <div>
          <h3>{movieName}</h3>
        </div>
      )}
      {!isTextInputMode && recognizedText && (
        <div>
          <h2>Recognized Text:</h2>
          <p>{recognizedText}</p>
        </div>
      )}
      {isTextInputMode && (
        <div className="text-input">
          <Form>
            <Form.Group className="mb-3" controlId="exampleForm.ControlInput1">
              <Form.Label>Enter Movie Dialog</Form.Label>
              <Form.Control className="input-field" type="text" placeholder="Avengers assemble" value={textInput}
            onChange={(e) => setTextInput(e.target.value)}/>
            </Form.Group>
          </Form>
          {/* <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
          /> */}
          <Button variant="outline-dark" onClick={handleTextInputSubmit}>Submit Text</Button>
        </div>
      )}
      <div className="switch-class">
        <Form>
          <Form.Check
            type="switch"
            id="custom-switch"
            label="Enable Text Input Mode"
            checked={isTextInputMode}
            onChange={(e) => setIsTextInputMode(e.target.checked)}
          />
        </Form>
        {/* <label>
          <input
            type="checkbox"
            checked={isTextInputMode}
            onChange={(e) => setIsTextInputMode(e.target.checked)}
          />
          Enable Text Input Mode
        </label> */}
      </div>
    </div>
  );
};

export default Slider;
