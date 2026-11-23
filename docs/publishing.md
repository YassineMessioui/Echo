# Publishing clips (#67)

Nothing is ever posted by `echo-clips run`. Publishing is three separate steps, and
only clips a human approved are posted:

```bash
echo-clips posts ep42 --llm-provider groq    # 1. a title, description and hashtags per clip
echo-clips review ep42 -i                    # 2. approve, reject or edit each clip
echo-clips publish ep42 --platform youtube   # 3. post the approved ones (private by default)
```

`clips/publish.json` keeps the text, the decisions and every post's link, so a re-run
never posts a clip twice and a failed clip is retried next time.

## YouTube Shorts: one-time setup

YouTube needs a Google Cloud project with an OAuth client for Echo (free, about 5 minutes):

1. **Project:** in the [Google Cloud console](https://console.cloud.google.com/), create a
   project (e.g. `echoclips`).
2. **API:** *APIs & Services → Library*, search **YouTube Data API v3**, click **Enable**.
3. **Consent screen:** *APIs & Services → OAuth consent screen* (or *Google Auth Platform*):
   app name `Echo`, your email as support and developer contact, audience **External**.
   Leave it in **Testing**, and under *Test users* add the Google account that owns the
   YouTube channel.
4. **Client:** *Credentials → Create credentials → OAuth client ID*, type **Desktop app**,
   name `echo-clips`. Download the JSON file.
5. **Sign in** (opens a browser once; Google warns that the app is unverified, which is
   expected for your own testing app: *Continue*):

   ```bash
   echo-clips auth youtube --client-secrets ~/Downloads/client_secret_XXXX.json
   ```

The client file and the refresh token live in `~/.config/echo/` (or `$ECHO_CONFIG_DIR`),
never in the repo or a job. To sign out, delete `~/.config/echo/youtube-token.json` and
remove Echo under your Google account's *Third-party access*.

### What gets uploaded

- Title, description and hashtags from `publish.json` (the `youtube` override if there is
  one); `#Shorts` is added to the description, and the hashtags also go in as tags.
- Visibility: `--privacy private` (default), `unlisted` or `public`.
- Echo's clips are vertical and at most 60 s, so YouTube files them as Shorts.

### Limits and errors

- **Quota:** a Cloud project gets a daily API quota (it resets at midnight Pacific time);
  when it's used up, `publish` stops with a clear message, keeping what was posted.
- **Sign-in expired or revoked:** run `echo-clips auth youtube` again. In *Testing* mode,
  Google expires refresh tokens after 7 days.
- **One clip rejected** (e.g. a title YouTube refuses): that clip's error is saved in
  `publish.json`, the other clips are still posted, and the exit code is 1.
