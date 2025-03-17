const express = require("express");
const { GoogleAuth } = require("google-auth-library");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors({
  origin: 'https://clima-lung.vercel.app',  // Replace with your Vercel app URL
}));
app.use(express.json());

async function getAccessToken() {
    const auth = new GoogleAuth({
        keyFile: "climalungbot-tkgt-111938f3b42c.json",  
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
