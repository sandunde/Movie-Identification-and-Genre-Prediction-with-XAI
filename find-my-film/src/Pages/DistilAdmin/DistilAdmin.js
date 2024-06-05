import React, { useState, useEffect } from "react";
import "./distilAdmin.css";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Spinner from "react-bootstrap/Spinner";
import * as d3 from "d3";

const DistilAdmin = () => {
  const [subtitle, setSubtitle] = useState("");
  const [files, setFiles] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);
  const [wordImportance, setWordImportance] = useState([]);
  const [loading, setLoading] = useState(false);

  const predict = async () => {
    setLoading(true);
    let formData = new FormData();
    formData.append("subtitle", subtitle);

    const response = await fetch("/predict-distil", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setPredictionResult(data);
    setWordImportance(data.attributions);
    setLoading(false);
  };

  const handleFileChange = (e) => {
    setFiles(e.target.files);
  };

  return (
    <div className="admin">
      <h1 className="admin-title">XAI Prediction</h1>
      <Row className="mb-3">
        <Col>
          <Form.Label>Enter Subtitle:</Form.Label>
          <div className="admin-subtitle">
            <Form.Control
              className="subtitle"
              as="textarea"
              rows={1}
              value={subtitle}
              onChange={(e) => setSubtitle(e.target.value)}
            />
          </div>
        </Col>
        <Col>
          <Form.Group controlId="formFile" className="mb-3">
            <Form.Label>Upload Files:</Form.Label>
            <Form.Control type="file" multiple onChange={handleFileChange} />
          </Form.Group>
        </Col>
      </Row>

      <Button className="predict-button" variant="outline-dark" onClick={predict}>
        Predict
      </Button>
      <div></div>
      {loading ? ( 
        <Spinner animation="border" role="status" className="sr-only">
          <span className="sr-only"></span>
        </Spinner>
      ) : (
        predictionResult && (
          <div className="prediction-result">
            <p>Predicted Genre: {predictionResult.predicted_genre}</p>
            {/* <p>Convergence Delta: {predictionResult.convergence_delta}</p> */}
            {/* <p>Attributions: {predictionResult.attributions.join(", ")}</p> */}
            {/* <p>Word Importance (Heatmap):</p>
            <Heatmap wordImportance={wordImportance} /> */}
          </div>
        )
      )}
    </div>
  );
};

// const Heatmap = ({ wordImportance }) => {
//   useEffect(() => {
//     if (wordImportance.length > 0) {
//       const width = 1200;
//       const height = 200;
//       const margin = { top: 20, right: 30, bottom: 90, left: 40 };

//       const svg = d3
//         .select("#heatmap")
//         .attr("width", width + margin.left + margin.right)
//         .attr("height", height + margin.top + margin.bottom)
//         .append("g")
//         .attr("transform", `translate(${margin.left}, ${margin.top})`);

//       const colorScale = d3
//         .scaleLinear()
//         .domain([0, Math.max(...wordImportance)])
//         .range(["lightblue", "red"]);

//       const xScale = d3
//         .scaleBand()
//         .domain(d3.range(wordImportance.length))
//         .range([0, width])
//         .padding(0.1);

//       svg
//         .append("g")
//         .attr("transform", `translate(0, ${height})`)
//         .call(d3.axisBottom(xScale));

//       svg
//         .selectAll("rect")
//         .data(wordImportance)
//         .enter()
//         .append("rect")
//         .attr("x", (d, i) => xScale(i))
//         .attr("y", 0)
//         .attr("width", xScale.bandwidth())
//         .attr("height", height)
//         .attr("fill", (d) => colorScale(d));

//       svg
//         .append("text")
//         .attr("x", width / 2)
//         .attr("y", height + margin.bottom / 3)
//         .attr("text-anchor", "middle")
//         .text("Light Blue (Low Importance) - Red (High Importance)")
//         .style("font-size", "14px")
//         .style("fill", "black");

//       svg
//         .append("text")
//         .attr("x", 0)
//         .attr("y", height + margin.bottom / 1.5)
//         .text("0: [CLS]")
//         .style("font-size", "14px")
//         .style("fill", "black");

//       svg
//         .append("text")
//         .attr("x", width - xScale.bandwidth())
//         .attr("y", height + margin.bottom / 1.5)
//         .text(`${wordImportance.length - 1}: [SEP]`)
//         .style("font-size", "14px")
//         .style("fill", "black");
//     }
//   }, [wordImportance]);

//   return <svg id="heatmap"></svg>;
// };

export default DistilAdmin;
