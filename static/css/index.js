let signUpbtn = document.getElementById("signUpbtn");
let Namefield = document.getElementById("Namefield");
let signInbtn = document.getElementById("signInbtn");
let title = document.getElementById("title");

signInbtn.onclick = function() {
    Namefield.style.maxHeight = "0px";
    title.innerHTML = "Sign In";
    signUpbtn.classList.add("active");
    signInbtn.classList.remove("active");
}

signUpbtn.onclick = function() {
    Namefield.style.maxHeight = "65px";
    title.innerHTML = "Sign Up";
    signInbtn.classList.remove("active");
    signUpbtn.classList.add("active");
}