// Display date and time
function displayDateTime() {
  const dateElem = document.getElementById("date");
  const timeElem = document.getElementById("time");
  setInterval(() => {
    const now = new Date();
    dateElem.textContent = now.toLocaleDateString();
    timeElem.textContent = now.toLocaleTimeString();
  })

}

// Function to format the duration in minutes and seconds
function formatDuration(duration) {
  const minutes = Math.floor(duration / 60);
  const seconds = duration % 60;
  return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}

// Update duration timer every second
function updateDuration() {
  const durationElem = document.getElementById("duration");

  let duration = 0;
  setInterval(function () {
    duration++;
    durationElem.textContent = formatDuration(duration);
  }, 1000);
}

//Function Call for Displaying Date, Time and Duration
displayDateTime();
updateDuration();