const express = require("express");
const { GoogleAuth } = require("google-auth-library");
const cors = require("cors");
require("dotenv").config();

const app = express();

//app.use(cors({ origin: 'http://localhost:3000' }));
app.use(cors());
app.use(express.json());

async function getAccessToken() {
    const auth = new GoogleAuth({
        //keyFile: "climalungbot-tkgt-111938f3b42c.json",  
        credentials: {
            "type": "service_account",
            "project_id": process.env.PROJECT_ID,
            "private_key_id": process.env.PRIVATE_KEY_ID,
            "private_key": process.env.PRIVATE_KEY.replace(/\\n/g, '\n'),
            "client_email": process.env.CLIENT_EMAIL,
            "client_id": process.env.CLIENT_ID,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": process.env.CLIENT_X509_CERT_URL,
        },
        scopes: ["https://www.googleapis.com/auth/dialogflow"],
    });

    const client = await auth.getClient();
    const token = await client.getAccessToken();
    return token.token;
}

app.get("/get-token", async (req, res) => {
    console.log('Received a request for token');
    try {
        const token = await getAccessToken();
        res.json({ token });
    } catch (error) {
        console.error('Error getting token:', error);
        res.status(500).json({ error: "Failed to get token", details: error.message });
    }
});

app.get("/download_models", (req, res) => {
    res.json({ message: "Downloading AQI Model Files..." });
  });

app.get("/", (req, res) => {
    res.send("Server is working!");
});

module.exports = app;

//app.listen(8000, () => console.log("Server running on port 8000"));
