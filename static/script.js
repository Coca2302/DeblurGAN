// static/script.js
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const inputImg = document.getElementById("input-img");
const outputImg = document.getElementById("output-img");
const statusDiv = document.getElementById("status");

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Chọn ảnh trước đã");
    return;
  }

  // show input preview
  inputImg.src = URL.createObjectURL(file);
  statusDiv.innerText = "Đang upload và xử lý...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json().catch(()=>null);
      statusDiv.innerText = "Lỗi: " + (err?.detail || res.statusText);
      return;
    }

    const blob = await res.blob();
    outputImg.src = URL.createObjectURL(blob);
    statusDiv.innerText = "Xong!";
  } catch (err) {
    console.error(err);
    statusDiv.innerText = "Lỗi kết nối với server.";
  }
});
