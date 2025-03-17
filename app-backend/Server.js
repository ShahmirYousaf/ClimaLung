const express = require("express");
const { GoogleAuth } = require("google-auth-library");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors({
  origin: 'https://clima-lung.vercel.app',  
}));
app.use(express.json());

async function getAccessToken() {
    const auth = new GoogleAuth({
        // keyFile: "climalungbot-tkgt-111938f3b42c.json",  
        credentials: {
            "type": "service_account",
            "project_id": process.env.project_id,
            "private_key_id": process.env.private_key_id,
            "private_key": process.env.private_key.replace(/\\n/g, '\n'),
            "client_email": process.env.client_email,
            "client_id": process.env.client_id,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": process.env.client_x509_cert_url
        },
        scopes: ["https://www.googleapis.com/auth/dialogflow"],
    });

    const client = await auth.getClient();
    const token = await client.getAccessToken();
    return token.token;
}

app.get("/get-token", async (req, res) => {
    try {
        const token = await getAccessToken();
        res.json({ token });
    } catch (error) {
        res.status(500).json({ error: "Failed to get token" });
    }
});

app.listen(8000, () => console.log("Server running on port 8000"));
