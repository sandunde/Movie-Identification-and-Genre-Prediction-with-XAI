import React from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./Signup.css";
import Logo from "../../Assets/logo.png";
import Facebook from "../../Assets/fb.png";
import Google from "../../Assets/g.png";
import Twitter from "../../Assets/x.png";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";

const Signup = () => {
    const navigate = useNavigate();
    const handleLogin = () =>{
        navigate("/home")
    }
    return (
        <div className="login-wrapper">
            <Row className="h-100">
                <Col sm={8} className="signup-background-col"></Col>
                <Col sm={4} className="right-container">
                    <div className="login-container">
                        <div className="logo-container">
                            <img
                                src={Logo}
                                style={{ width: "50px", height: "50px" }}
                                alt="Logo"
                            />
                            <h3>FindMyFilm</h3>
                        </div>
                        <h5>Welcome to FindMyFilm! 👋</h5>
                        <p style={{ fontSize: "12px" }}>
                            Create a free account and start the <br /> adventure
                        </p>
                        <div className="form">
                            <Form>
                                <Form.Group className="mb-3">
                                    <Form.Control
                                        type="text"
                                        placeholder="Name"
                                        className="custom-width-input"
                                    />
                                </Form.Group>
                                <Form.Group className="mb-3">
                                    <Form.Control
                                        type="email"
                                        placeholder="Email"
                                        className="custom-width-input"
                                    />
                                </Form.Group>
                                <Form.Group className="mb-3">
                                    <Form.Control
                                        type="password"
                                        placeholder="Password"
                                        className="custom-width-input"
                                    />
                                </Form.Group>
                                <Button
                                    variant="primary"
                                    type="submit"
                                    className="custom-width-input"
                                    onClick={handleLogin}
                                >
                                    SIGN UP
                                </Button>
                            </Form>
                            <p style={{ marginTop: "10px" }}>
                                Have an account?{" "}
                                <Link to="/" style={{ textDecoration: "none", marginLeft: "120px" }}>
                                    Login
                                </Link>
                            </p>
                            <p style={{ marginTop: "-10px" }}>
                                Admin User?{" "}
                                <Link
                                    to="/admin-login"
                                    style={{ textDecoration: "none", marginLeft: "70px" }}
                                >
                                    Admin Dashboard
                                </Link>
                            </p>
                        </div>
                        <div className="login-footer">
                            <div className="or-line">
                                <hr style={{ width: "45%" }} />
                                <p style={{ margin: "5px" }}>Or</p>
                                <hr style={{ width: "45%" }} />
                            </div>
                            <div className="footer-logo">
                                <img src={Facebook} alt="facebook" />
                                <img src={Google} alt="google"/>
                                <img src={Twitter} alt="twitter"/>
                            </div>
                        </div>
                    </div>
                </Col>
            </Row>
        </div>
    );
};

export default Signup;
