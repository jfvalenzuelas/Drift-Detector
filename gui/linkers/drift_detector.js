let {PythonShell} = require("python-shell");
const path = require("path");
const $ = require("jquery");
const swal = require("sweetalert");

function detect_drift() {
    var filePath = $("#formControlOpenFile").val();
    var language = $("#selectLanguage").val();
    var windows = $("#windowsSelect").val();
    var keywords = $("#keywordsSelect").val();

    var options = {
        scriptPath: path.join(__dirname, '/../engine/'),
        args: [filePath, language, windows, keywords]
    }

    let drift_detector = new PythonShell('drift_detector.py', options);

    drift_detector.on('message', function(message) {
        swal(message);
    });
}