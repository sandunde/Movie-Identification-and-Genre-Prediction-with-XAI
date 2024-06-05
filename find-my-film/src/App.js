import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import Login from "./Components/Login/Login";
import AdminLogin from "./Components/Login/AdminLogin";
import Signup from "./Components/Login/Signup";
import Home from "./Pages/Home/Home";
import Result from "./Pages/Result/Result";
import Admin from "./Pages/Admin/Admin";
import DistilAdmin from "./Pages/DistilAdmin/DistilAdmin";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/admin-login" element={<AdminLogin />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/home" element={<Home />} />
          <Route path="/result" element={<Result />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/distil-admin" element={<DistilAdmin />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
