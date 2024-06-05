import React, { useState } from "react";
import "./Navbar.css";

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);

    const toggleMenu = () => {
        setIsOpen(!isOpen);
    };

    return (
            <nav className="navbar-container">
                <div className="nav-logo">Logo</div>
                <ul className={isOpen ? "nav-links open" : "nav-links"}>
                    <li>
                        <a href="/">Home</a>
                    </li>
                    <li>
                        <a href="/about">About</a>
                    </li>
                    <li>
                        <a href="/services">Services</a>
                    </li>
                    <li>
                        <a href="/contact">Contact</a>
                    </li>
                </ul>
                <div className="burger" onClick={toggleMenu}>
                    <div className={isOpen ? "line1 open" : "line1"}></div>
                    <div className={isOpen ? "line2 open" : "line2"}></div>
                    <div className={isOpen ? "line3 open" : "line3"}></div>
                </div>
            </nav>
    );
};

export default Navbar;
