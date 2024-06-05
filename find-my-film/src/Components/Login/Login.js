import React, {useState} from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./Login.css";
import Logo from "../../Assets/logo.png";
import Facebook from "../../Assets/fb.png";
import Google from "../../Assets/g.png";
import Twitter from "../../Assets/x.png";
import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";

const Login = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const handleLogin = () => {
        navigate("/home");
    };
    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    }
    const handlePasswordChange = (event) => {
        setPassword(event.target.value);
    }

    const isDisabled = (email.trim() === '' && password.trim() === '');

    return (
        <div className="login-wrapper">
            <Row className="h-100">
                <Col sm={8} className="background-col"></Col>
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
                            Please sign in to your account and start the <br /> adventure
                        </p>
                        <div className="form">
                            <Form>
                                <Form.Group className="mb-3">
                                    <Form.Control
                                        type="email"
                                        placeholder="Email"
                                        className="custom-width-input"
                                        onChange={handleEmailChange}
                                    />
                                </Form.Group>
                                <Form.Group className="mb-3">
                                    <Form.Control
                                        type="password"
                                        placeholder="Password"
                                        className="custom-width-input"
                                        onChange={handlePasswordChange}
                                    />
                                </Form.Group>
                                <Form.Group className="mb-3" controlId="formBasicCheckbox">
                                    <div className="forgot-password">
                                        <Form.Check type="checkbox" label="Remember Me" />
                                        <a href="/">Forgot Password?</a>
                                    </div>
                                </Form.Group>
                                <Button
                                    variant="primary"
                                    type="submit"
                                    className="custom-width-input"
                                    onClick={handleLogin}
                                    disabled={isDisabled}
                                >
                                    SIGN IN
                                </Button>
                            </Form>
                            <p style={{ marginTop: "10px" }}>
                                New on our platform?{" "}
                                <Link to="/signup" style={{ textDecoration: "none" }}>
                                    Create an account
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
                                <img src={Google} alt="google" />
                                <img src={Twitter} alt="twitter" />
                            </div>
                        </div>
                    </div>
                </Col>
            </Row>
        </div>
    );
};

export default Login;
